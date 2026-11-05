"""Phylogenetic topology recovery: Vectorized NJ with SQS-based NNI refinement."""
import math
import time
import numpy as np
from itertools import combinations

_START_TIME = time.time()
_DEADLINE = _START_TIME + 250.0  # Safety margin


def _jc_distance_matrix(sequences, labels):
    """Compute pairwise Jukes-Cantor corrected distances using NumPy."""
    n = len(labels)
    L = len(sequences[labels[0]])
    
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq_arr = np.array([[mapping[c] for c in sequences[lab]] for lab in labels], dtype=np.int8)
    
    # Count differences
    diff = (seq_arr[:, np.newaxis, :] != seq_arr[np.newaxis, :, :]).sum(axis=-1)
    
    p = diff.astype(np.float64) / L
    p = np.clip(p, 0, 0.7499)
    D = -0.75 * np.log(1.0 - (4.0 / 3.0) * p)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) * 0.5
    
    return D


def _vectorized_nj(D, labels):
    """
    Neighbor Joining using NumPy for speed.
    Returns a list of tuples representing the tree structure: (left_child, right_child, label)
    """
    n = len(labels)
    if n <= 1:
        return [None]
    if n == 2:
        return [(0, 1, None)]
    
    # Active nodes: 0..n-1
    active = np.arange(n)
    # Distances to active nodes
    dist = D.copy()
    
    # Newick strings for each node
    newick = [labels[i] for i in range(n)]
    
    next_id = n
    tree_edges = [] # Store (parent, left, right)
    
    while len(active) > 3:
        if time.time() > _DEADLINE - 5.0:
            break
            
        m = len(active)
        
        # Compute r vector: sum of distances from each active node to all other active nodes
        # dist is full n x n matrix, but we only care about active nodes
        # We can just sum the rows of dist[active][:, active]
        r = dist[active][:, active].sum(axis=1)
        
        # Q matrix: (m-2)*D_ab - r_a - r_b
        D_active = dist[active][:, active]
        Q = (m - 2) * D_active - r[:, np.newaxis] - r[np.newaxis, :]
        np.fill_diagonal(Q, np.inf)
        
        # Find min Q
        min_val = np.min(Q)
        min_idx = np.unravel_index(np.argmin(Q), Q.shape)
        a_idx, b_idx = min_idx
        a = active[a_idx]
        b = active[b_idx]
        
        # Create new node
        new_id = next_id
        next_id += 1
        
        # Newick for new node
        new_newick = f"({newick[a]},{newick[b]})"
        
        # Update distances
        # New distance to x: 0.5 * (D_ax + D_bx - D_ab)
        # x ranges over active, excluding a and b
        others = np.setdiff1d(active, [a, b])
        if len(others) > 0:
            D_ab = D[a, b]
            new_dists = 0.5 * (dist[a, others] + dist[b, others] - D_ab)
            # Update dist matrix
            dist[new_id, others] = new_dists
            dist[others, new_id] = new_dists
            
        # Remove a, b from active, add new_id
        active = np.append(np.setdiff1d(active, [a, b]), new_id)
        
        # Store edge
        tree_edges.append((new_id, a, b))
        
        # Update newick list (need to extend it if new_id > len(newick)-1)
        # Actually, new_id is just an integer index. We can use a dictionary or list.
        # Since new_id increments, we can just append to newick list.
        while len(newick) <= new_id:
            newick.append("")
        newick[new_id] = new_newick
        
    # Final 3 nodes
    if len(active) == 3:
        a, b, c = active
        # Connect them: (a, (b, c))
        new_id = next_id
        next_id += 1
        tree_edges.append((new_id, b, c))
        tree_edges.append((new_id, a, new_id)) # Wait, structure is (a, (b,c))
        # Let's just construct the final Newick string directly
        final_newick = f"({newick[a]},{newick[b]},{newick[c]})"
        # But we need to return a structure that can be refined.
        # Let's return the newick string and let the refiner parse it.
        return final_newick
    else:
        # Should not happen
        return "(" + ",".join(newick[x] for x in active) + ")"


def _parse_newick(newick_str):
    """
    Parse Newick string into a tree structure.
    Returns:
    - leaves: list of leaf labels
    - internal_nodes: list of internal node indices
    - children: dict mapping node_id -> (left_child, right_child)
    - leaf_map: dict mapping leaf_label -> node_id
    - node_to_leaves: dict mapping node_id -> set of leaf labels
    """
    leaves = []
    leaf_map = {}
    children = {}
    node_to_leaves = {}
    
    # Stack-based parser
    # Node types: 'leaf' or 'internal'
    # We assign IDs: 0..n-1 for leaves, n.. for internal
    
    current_id = 0
    leaf_count = 0
    internal_count = 0
    
    # We need to build the tree from the string.
    # This is a bit complex. Let's use a recursive descent parser or stack.
    
    # Let's use a simpler approach: just find the splits directly?
    # No, for NNI we need the tree structure.
    
    # Let's implement a stack parser.
    # Elements on stack: (node_id, state)
    # state: 0 = expecting left child, 1 = expecting right child, 2 = done
    
    # Actually, let's just use the Newick string manipulation for NNI if we can identify the subtrees.
    # But extracting leaf sets is crucial.
    
    # Let's try a different representation: 
    # The tree is defined by its splits.
    # For NNI, we need to find internal edges.
    
    # Let's stick to the Newick string and use a helper to get leaf sets for subtrees.
    pass

def _get_subtree_leaves(newick_str, start, end):
    """
    Get the set of leaf labels in the subtree defined by newick_str[start:end].
    Assumes the substring is a valid Newick subtree (possibly without outer parens if it's a leaf).
    """
    leaves = set()
    i = 0
    s = newick_str
    n = len(s)
    
    while i < n:
        if s[i] == '(':
            # Find matching ')'
            depth = 0
            j = i
            while j < n:
                if s[j] == '(':
                    depth += 1
                elif s[j] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            # Recurse
            sub_leaves = _get_subtree_leaves(s[i+1:j])
            leaves.update(sub_leaves)
            i = j + 1
        elif s[i] == ',':
            i += 1
        elif s[i] == ':':
            # Skip branch length
            while i < n and s[i] not in ',)':
                i += 1
        else:
            # Leaf label
            j = i
            while j < n and s[j] not in ',):':
                j += 1
            label = s[i:j]
            if label:
                leaves.add(label)
            i = j
            
    return leaves

def _nni_refine_sqs(D, labels, newick_str):
    """
    NNI refinement using Sum of Quartet Scores (SQS).
    """
    n = len(labels)
    if n < 4:
        return newick_str
        
    label_set = set(labels)
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    
    # Precompute Q scores for all quartets? 
    # Or compute on the fly.
    
    # Helper to compute SQS for a specific NNI move
    # Subtrees: A, B, C, D (sets of labels)
    # Current: (A,B)|(C,D)
    # Alt 1: (A,C)|(B,D)
    # Alt 2: (A,D)|(B,C)
    
    def sqs_change(A, B, C, D, current_topo):
        """
        Compute the change in SQS if we switch from current_topo to the best alternative.
        current_topo: 'ab|cd', 'ac|bd', 'ad|bc'
        Returns: (best_topo, delta_sqs)
        """
        # We need to sum Q for all quartets a in A, b in B, c in C, d in D
        # Q_ab_cd = D_ab + D_cd - D_ac - D_ad - D_bc - D_bd
        
        # To make it fast, we can precompute sum of D between sets.
        # sum_D(X, Y) = sum_{x in X, y in Y} D[x, y]
        
        # We will compute the SQS contribution for the three topologies.
        # SQS_topo = sum_{quartets} Q_topo
        
        # Let's compute the total Q for each topology.
        
        # Indices
        A_idx = [label_to_idx[l] for l in A]
        B_idx = [label_to_idx[l] for l in B]
        C_idx = [label_to_idx[l] for l in C]
        D_idx = [label_to_idx[l] for l in D]
        
        # Convert to numpy arrays
        A_arr = np.array(A_idx)
        B_arr = np.array(B_idx)
        C_arr = np.array(C_idx)
        D_arr = np.array(D_idx)
        
        # D_ab is D[A_arr[:, None], B_arr[None, :]]
        # Sum of D_ab: D[A_arr, B_arr].sum()
        
        sum_ab = D[A_arr[:, None], B_arr[None, :]].sum()
        sum_cd = D[C_arr[:, None], D_arr[None, :]].sum()
        sum_ac = D[A_arr[:, None], C_arr[None, :]].sum()
        sum_ad = D[A_arr[:, None], D_arr[None, :]].sum()
        sum_bc = D[B_arr[:, None], C_arr[None, :]].sum()
        sum_bd = D[B_arr[:, None], D_arr[None, :]].sum()
        
        # Q_ab_cd = D_ab + D_cd - D_ac - D_ad - D_bc - D_bd
        # Sum over all quartets:
        # n_quartets = len(A)*len(B)*len(C)*len(D)
        
        n_a = len(A_arr)
        n_b = len(B_arr)
        n_c = len(C_arr)
        n_d = len(D_arr)
        
        # Sum of D_ab over all quartets: each D_ab is counted n_c * n_d times
        total_ab = sum_ab * n_c * n_d
        total_cd = sum_cd * n_a * n_b
        total_ac = sum_ac * n_b * n_d
        total_ad = sum_ad * n_b * n_c
        total_bc = sum_bc * n_a * n_d
        total_bd = sum_bd * n_a * n_c
        
        sqs_ab_cd = total_ab + total_cd - total_ac - total_ad - total_bc - total_bd
        sqs_ac_bd = total_ac + total_bd - total_ab - total_cd - total_ad - total_bc
        sqs_ad_bc = total_ad + total_bc - total_ab - total_cd - total_ac - total_bd
        
        scores = {
            'ab|cd': sqs_ab_cd,
            'ac|bd': sqs_ac_bd,
            'ad|bc': sqs_ad_bc
        }
        
        # Find best
        best_topo = min(scores, key=scores.get)
        best_score = scores[best_topo]
        
        # Delta is current - best (we want to minimize SQS, so if best < current, delta is negative improvement)
        # Wait, SQS is sum of Q. We want to minimize Q? 
        # Q = D_ab + D_cd - ...
        # If the tree is correct, D_ab and D_cd are small, D_ac etc are large.
        # So Q_ab_cd should be negative.
        # Minimizing SQS (making it more negative) is good.
        
        current_score = scores[current_topo]
        delta = best_score - current_score
        
        return best_topo, delta

    newick = newick_str
    improved = True
    while improved:
        if time.time() > _DEADLINE - 10.0:
            break
            
        improved = False
        
        # Find all internal edges
        # Pattern: ... (X, Y) , (Z, W) ...
        # We look for "),("
        
        positions = []
        for pos in range(len(newick) - 3):
            if newick[pos:pos+2] == ")," and newick[pos+2:pos+3] == "(":
                positions.append(pos)
                
        for pos in positions:
            if time.time() > _DEADLINE - 5.0:
                break
                
            # Extract left subtree (X, Y) and right subtree (Z, W)
            # Left ends at pos
            left_end = pos
            depth = 0
            left_start = left_end
            for k in range(left_end, -1, -1):
                if newick[k] == ')':
                    depth += 1
                elif newick[k] == '(':
                    depth -= 1
                    if depth == 0:
                        left_start = k
                        break
                        
            # Right starts at pos+2
            right_start = pos + 2
            depth = 0
            right_end = right_start
            for k in range(right_start, len(newick)):
                if newick[k] == '(':
                    depth += 1
                elif newick[k] == ')':
                    depth -= 1
                    if depth == 0:
                        right_end = k
                        break
                        
            # Left content: newick[left_start+1 : left_end]
            # This is "X,Y"
            left_content = newick[left_start+1:left_end]
            # Right content: newick[right_start+1 : right_end]
            # This is "Z,W"
            right_content = newick[right_start+1:right_end]
            
            # Split left_content into X and Y
            depth = 0
            comma_pos = -1
            for k in range(len(left_content)):
                if left_content[k] == '(':
                    depth += 1
                elif left_content[k] == ')':
                    depth -= 1
                elif left_content[k] == ',' and depth == 0:
                    comma_pos = k
                    break
                    
            if comma_pos == -1:
                continue
                
            X_str = left_content[:comma_pos]
            Y_str = left_content[comma_pos+1:]
            
            # Split right_content into Z and W
            depth = 0
            comma_pos2 = -1
            for k in range(len(right_content)):
                if right_content[k] == '(':
                    depth += 1
                elif right_content[k] == ')':
                    depth -= 1
                elif right_content[k] == ',' and depth == 0:
                    comma_pos2 = k
                    break
                    
            if comma_pos2 == -1:
                continue
                
            Z_str = right_content[:comma_pos2]
            W_str = right_content[comma_pos2+1:]
            
            # Get leaf sets
            set_X = _get_subtree_leaves(X_str)
            set_Y = _get_subtree_leaves(Y_str)
            set_Z = _get_subtree_leaves(Z_str)
            set_W = _get_subtree_leaves(W_str)
            
            if not set_X or not set_Y or not set_Z or not set_W:
                continue
            
            # Current topology: (X, Y) | (Z, W) -> 'ab|cd' where A=X, B=Y, C=Z, D=W
            best_topo, delta = sqs_change(set_X, set_Y, set_Z, set_W, 'ab|cd')
            
            if best_topo != 'ab|cd' and delta < -1e-9:
                # Perform NNI move
                if best_topo == 'ac|bd':
                    # New: (X, Z) | (Y, W)
                    new_left = f"({X_str},{Z_str})"
                    new_right = f"({Y_str},{W_str})"
                else: # 'ad|bc'
                    # New: (X, W) | (Y, Z)
                    new_left = f"({X_str},{W_str})"
                    new_right = f"({Y_str},{Z_str})"
                    
                old_segment = f"({left_content},{right_content})"
                new_segment = f"({new_left},{new_right})"
                
                if old_segment in newick:
                    newick = newick.replace(old_segment, new_segment, 1)
                    improved = True
                    break # Restart scan
                    
    return newick

def reconstruct(sequences):
    global _START_TIME, _DEADLINE
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 1:
        return labels[0] + ";" if labels else ""
    if n == 2:
        return f"({labels[0]},{labels[1]});"
    if n == 3:
        return f"({labels[0]},{labels[1]},{labels[2]});"
    
    try:
        # 1. Distance Matrix
        D = _jc_distance_matrix(sequences, labels)
        
        # 2. NJ
        newick = _vectorized_nj(D, labels)
        
        # 3. NNI Refinement with SQS
        if time.time() < _DEADLINE - 10.0:
            newick = _nni_refine_sqs(D, labels, newick)
        
        # Ensure the Newick ends with ';'
        if not newick.endswith(';'):
            newick += ';'
        
        return newick
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return "(" + ",".join(labels) + ");"

# EVOLVE-BLOCK-END
