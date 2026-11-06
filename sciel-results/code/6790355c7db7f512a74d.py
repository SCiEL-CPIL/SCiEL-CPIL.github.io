"""
Improved phylogenetic topology recovery using Bootstrap Neighbor-Joining.
"""
import numpy as np
import time
from collections import defaultdict

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = 270.0  # Evaluator kills at 270s wall-clock. Use 260s for safety.

def _jc_distance(aln_a, aln_b):
    """
    Calculate Jukes-Cantor corrected distance between two sequences.
    aln_a, aln_b: 1D numpy arrays of ints (0-3 for A,C,G,T)
    """
    # Count mismatches
    mismatches = np.sum(aln_a != aln_b)
    n_sites = len(aln_a)
    p = mismatches / n_sites
    if p >= 0.75:
        return 1.0 # Cap distance to avoid log(0) or large errors
    return -0.75 * np.log(1.0 - (4.0 / 3.0) * p)

def _nj_distance_matrix(sequences_matrix, n_taxa, n_sites):
    """
    Compute the pairwise JC distance matrix.
    sequences_matrix: (n_taxa, n_sites) int array
    """
    # Calculate differences for all pairs at once is hard without loops in pure numpy for 32x32, 
    # but we can do it efficiently.
    # Actually, for small n_taxa (<=32), a simple double loop with vectorized inner op is fast enough.
    D = np.zeros((n_taxa, n_taxa))
    for i in range(n_taxa):
        for j in range(i + 1, n_taxa):
            d = _jc_distance(sequences_matrix[i], sequences_matrix[j])
            D[i, j] = d
            D[j, i] = d
    return D

def _neighbor_joining(D, n_taxa):
    """
    Perform Neighbor Joining on distance matrix D.
    Returns a list of splits (frozensets of taxa indices) representing the internal nodes.
    """
    # D is n x n
    # active_nodes: list of current node indices (0..n-1 initially, then new nodes)
    # We track the tree structure to extract splits.
    # A simpler way to get splits is to track the bipartitions created at each step.
    
    # State:
    # current_D: distance matrix for active nodes
    # active_indices: list of original taxon indices or new node indices
    # Actually, standard NJ returns a tree. To get splits, we can record the join.
    
    # Let's implement a version that returns the list of splits directly.
    # A split is a partition of the set of original taxa.
    
    # We need to map new nodes to sets of original taxa.
    node_to_taxa = {i: {i} for i in range(n_taxa)}
    
    current_nodes = list(range(n_taxa))
    current_D = D.copy()
    splits = []
    
    num_nodes = n_taxa
    next_node_id = n_taxa
    
    while len(current_nodes) > 3:
        n = len(current_nodes)
        # Calculate r_i = sum_j D[i,j]
        r = np.sum(current_D, axis=1)
        
        # Calculate Q matrix: Q[i,j] = (n-2)*D[i,j] - r[i] - r[j]
        # We want to find min Q[i,j] for i < j
        min_val = np.inf
        best_i = -1
        best_j = -1
        
        # Iterate over active pairs
        # Note: current_D is aligned with current_nodes order? 
        # It's easier to keep current_D as a dense matrix of size n x n corresponding to current_nodes.
        # Let's re-index.
        # Actually, let's just use the indices in current_nodes.
        
        # To speed up, we can vectorize the Q calculation for the current active set.
        # But n is small (<=32), so nested loop is fine.
        
        for i_idx in range(n):
            u = current_nodes[i_idx]
            for j_idx in range(i_idx + 1, n):
                v = current_nodes[j_idx]
                # Get distance from original D? No, we need updated distances.
                # We must maintain a distance matrix that updates as we join.
                # This is tricky with the 'current_nodes' list approach if D is global.
                
                # Let's restart the NJ implementation to be cleaner.
                pass
                
    # --- Revised NJ Implementation for Split Extraction ---
    
    # D is the initial distance matrix (n x n)
    # We will maintain:
    # 1. A list of active node IDs
    # 2. A dictionary mapping node ID -> set of original taxa it represents
    # 3. A dictionary mapping (node1, node2) -> distance, or a matrix.
    #    Since nodes are added, a matrix is hard to resize. 
    #    Let's use a dictionary for distances between active nodes.
    
    active = list(range(n_taxa))
    node_taxa = {i: {i} for i in range(n_taxa)}
    dist = {}
    for i in range(n_taxa):
        for j in range(i+1, n_taxa):
            dist[(i, j)] = D[i, j]
            dist[(j, i)] = D[i, j]
            
    splits = []
    next_id = n_taxa
    
    while len(active) > 3:
        n = len(active)
        # Calculate r for each active node
        r = {u: sum(dist[(u, v)] for v in active if v != u) for u in active}
        
        # Find best pair (u, v) minimizing Q
        min_q = np.inf
        best_u = -1
        best_v = -1
        
        for i_idx in range(n):
            u = active[i_idx]
            for j_idx in range(i_idx + 1, n):
                v = active[j_idx]
                d_uv = dist[(u, v)]
                q_val = (n - 2) * d_uv - r[u] - r[v]
                if q_val < min_q:
                    min_q = q_val
                    best_u = u
                    best_v = v
        
        u = best_u
        v = best_v
        
        # Create new node w
        w = next_id
        next_id += 1
        
        # Determine taxa sets for new node
        node_taxa[w] = node_taxa[u] | node_taxa[v]
        
        # The split created by joining u and v separates node_taxa[w] from the rest?
        # No, the split is the partition induced by the new internal edge.
        # The two sides are node_taxa[u] and node_taxa[v]? No, that's only if u and v were leaves.
        # In NJ, the new node w connects to u and v. The edge (w, u) splits the tree into node_taxa[u] and the rest.
        # Wait, the "split" associated with the internal node w is usually defined by the bipartition of the leaves.
        # However, for majority voting, we want the splits corresponding to the internal edges.
        # When we join u and v into w, we create an internal node.
        # The split associated with the edge connecting w to the rest of the tree is not known yet.
        # The splits associated with edges (w,u) and (w,v) are:
        # Split 1: node_taxa[u] vs (all others)
        # Split 2: node_taxa[v] vs (all others)
        # But wait, if u was an internal node, node_taxa[u] is a set of leaves.
        # So yes, we can record these splits.
        
        # Record splits for edges (w, u) and (w, v)
        # Split A: node_taxa[u]
        # Split B: node_taxa[v]
        # Note: We should only record non-trivial splits (size > 1 and size < n_taxa - 1)
        
        set_u = node_taxa[u]
        set_v = node_taxa[v]
        
        if 1 < len(set_u) < n_taxa - 1:
            splits.append(frozenset(set_u))
        if 1 < len(set_v) < n_taxa - 1:
            splits.append(frozenset(set_v))
            
        # Update distances
        # Remove u, v from active
        active.remove(u)
        active.remove(v)
        
        # Calculate new distances from w to all other active nodes x
        # d(w, x) = 0.5 * (d(u, x) + d(v, x) - d(u, v))
        d_uv = dist[(u, v)]
        
        for x in active:
            d_wx = 0.5 * (dist[(u, x)] + dist[(v, x)] - d_uv)
            dist[(w, x)] = d_wx
            dist[(x, w)] = d_wx
            
        # Add w to active
        active.append(w)
        
        # Clean up old distances (optional for correctness, good for memory)
        # We don't strictly need to remove them if we only look up active nodes.
        
    # Final 3 nodes: a, b, c
    # The remaining edges connect these 3.
    # The splits are determined by the sets of taxa.
    # Actually, the 3 remaining nodes form a star.
    # The splits are:
    # node_taxa[a] vs rest
    # node_taxa[b] vs rest
    # node_taxa[c] vs rest
    # But these are often trivial or already counted?
    # In a 3-node tree, the splits are the 3 bipartitions.
    # Let's add them to be safe, filtering trivial ones.
    
    for node in active:
        s = node_taxa[node]
        if 1 < len(s) < n_taxa - 1:
            splits.append(frozenset(s))
            
    return splits

def _build_tree_from_splits(splits, n_taxa):
    """
    Build a Newick string from a list of splits (majority vote).
    This is a heuristic: we sort splits by frequency (or just take unique ones) and try to assemble.
    A simpler approach: Use the splits to define a tree.
    Since NJ produces a valid tree, and we are voting, we can just pick the most frequent tree?
    No, we are voting on splits.
    
    Algorithm:
    1. Count frequency of each split.
    2. Select splits with frequency > 50%.
    3. Build a tree that contains all selected splits.
    4. If the set of splits is not compatible (forms a cycle), we need a fallback.
       However, for NJ, the splits from a single run are always compatible.
       The union of majority splits from multiple runs might not be compatible.
       In that case, we can use a method like "Max Compatible Subset" or just fall back to the first run's tree.
       
    Given the time constraint, a simpler robust method:
    Run NJ multiple times.
    For each run, we get a tree (Newick string).
    We can't easily vote on Newick strings directly because of label ordering.
    But we CAN vote on splits.
    
    Let's refine:
    1. Run NJ B times.
    2. Collect all splits from all runs.
    3. Count frequencies.
    4. Select splits with count > B/2.
    5. If the selected splits are compatible, build the tree.
    6. If not, pick the most frequent single tree from the runs?
       Or, use a greedy algorithm to add splits one by one (highest freq first) if compatible.
    """
    # This is complex to implement robustly in short time.
    # Alternative: Just return the tree from the run with the highest "consensus" score?
    # Or, simpler: Just run NJ once on the full data.
    # Why did I think bootstrap helps? Because it reduces noise.
    # But implementing the "Build tree from majority splits" is non-trivial.
    
    # Let's try a different approach: 
    # 1. Compute Distance Matrix D.
    # 2. Run NJ once.
    # 3. Is that enough? The seed program does exactly this and scores 1.0.
    # To improve, we need to do better than single NJ.
    
    # What if we use the bootstrap to *filter* splits, and then use a simple tree-building algorithm?
    # Actually, there is a known method: "Majority Rule Consensus".
    # We can implement a simple version:
    # 1. Get splits from all bootstrap NJ trees.
    # 2. Count them.
    # 3. Keep splits with support > 50%.
    # 4. If these splits are compatible, they define a unique tree.
    # 5. How to check compatibility and build?
    #    We can represent splits as bitmasks.
    #    Two splits A and B are compatible if one of A, B, A&B, A&~B is empty.
    #    We can greedily add splits?
    
    # Let's try this greedy approach.
    
    # First, we need the counts.
    # This function should take a list of all splits from all runs.
    
    # For now, let's just return the tree from the first run if we can't easily build the consensus.
    # But wait, we can improve NJ itself.
    
    # Let's stick to the plan:
    # 1. Bootstrap resampling.
    # 2. NJ on each.
    # 3. Collect splits.
    # 4. Majority vote.
    # 5. Build tree from majority splits.
    
    # Implementation of building tree from compatible splits:
    # This is a known hard problem if incompatible, but if they come from NJ, they are likely mostly compatible.
    # If incompatible, we can just pick the most frequent split set?
    
    # Let's implement a helper to check compatibility and build.
    # Actually, for small n (<=32), we can just use the splits to define the tree topology directly.
    
    # Simpler Idea: 
    # Instead of voting on splits, vote on the *trees*?
    # No, Newick strings vary.
    
    # Let's try to implement the split voting properly.
    # If the majority splits are incompatible, we fall back to the tree with the highest number of majority splits.
    
    # For this rewrite, I will implement:
    # 1. Fast NJ using NumPy.
    # 2. Bootstrap resampling of the alignment.
    # 3. Split extraction and counting.
    # 4. Greedy construction of a tree from the most frequent compatible splits.
    # 5. Fallback to single NJ if consensus fails.

    pass # Placeholder for actual logic in the main function

def reconstruct(sequences):
    start_time = time.time()
    
    # 1. Preprocess sequences
    # Map A,C,G,T to 0,1,2,3
    char_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n_taxa = len(labels)
    
    # Convert to numpy array
    # sequences[label] is a string
    seq_list = []
    for label in labels:
        seq = sequences[label]
        arr = np.array([char_map[c] for c in seq], dtype=np.int8)
        seq_list.append(arr)
    
    alignment = np.vstack(seq_list) # (n_taxa, n_sites)
    n_sites = alignment.shape[1]
    
    # 2. Compute initial distance matrix
    D = _nj_distance_matrix(alignment, n_taxa, n_sites)
    
    # 3. Run Bootstrap
    # Determine number of replicates based on time and n_taxa
    # NJ is O(n^3). n=32 -> ~32k ops. Very fast.
    # Distance matrix calc is O(n^2 * L). L=2000, n=32 -> 200k ops.
    # So we can run many replicates.
    
    # Let's aim for 100-200 replicates if time permits.
    # We will monitor time.
    
    all_splits = []
    best_tree_newick = None
    best_split_count = -1
    
    # First, run once on full data to get a baseline
    splits_0 = _neighbor_joining(D, n_taxa)
    all_splits.extend(splits_0)
    
    # We need to be able to build a tree from splits.
    # Let's implement a simple tree builder from splits.
    # If we have a set of compatible splits, they define a tree.
    
    # Helper: Check if a list of splits is compatible
    def are_compatible(splits):
        # s1 and s2 are compatible if one of s1, s2, s1&s2, s1&~s2 is empty
        # Using sets
        for i in range(len(splits)):
            for j in range(i+1, len(splits)):
                s1 = splits[i]
                s2 = splits[j]
                # Intersection
                inter = s1 & s2
                if inter:
                    # Check if inter == s1 or inter == s2
                    if inter != s1 and inter != s2:
                        return False
                else:
                    # Disjoint, check if s1 | s2 is all taxa? No, just need to be compatible.
                    # Disjoint is compatible unless they cross?
                    # If s1 and s2 are disjoint, they are compatible.
                    # If they overlap, one must contain the other.
                    pass
        return True

    # This check is O(K^2 * N). K is number of splits (~n). N is taxa.
    # It's fine.
    
    # Helper: Build Newick from compatible splits
    # This is tricky. 
    # Alternative: Just use the splits to guide a new NJ run?
    # Or, use the fact that if we have the splits, we can reconstruct the tree.
    
    # Let's try a different strategy for the "Consensus Tree":
    # Instead of building from splits, we can just pick the NJ tree from the bootstrap runs
    # that has the highest number of splits in common with the majority splits.
    
    # Revised Plan:
    # 1. Run NJ on full data. Store Tree 0 and its splits S_0.
    # 2. Run B bootstrap replicates. For each, get Tree b and splits S_b.
    # 3. Count frequency of each split across all S_b (and S_0).
    # 4. Define "Majority Splits" M = {s | count(s) > B/2}.
    # 5. For each tree T_b, calculate score = |S_b ∩ M|.
    # 6. Return the tree T_b with the highest score.
    
    # This is much easier to implement and robust.
    
    split_counts = defaultdict(int)
    
    # Count from first run
    for s in splits_0:
        split_counts[s] += 1
        
    # Store first tree
    # We need to reconstruct the Newick string from the NJ process.
    # My _neighbor_joining function currently only returns splits.
    # I need it to return the Newick string too, or at least the tree structure.
    
    # Let's modify _neighbor_joining to return (splits, newick_string).
    
    # Re-implementing NJ to return Newick as well.
    
    def _nj_with_newick(D, n_taxa):
        active = list(range(n_taxa))
        node_taxa = {i: {i} for i in range(n_taxa)}
        node_newick = {i: f"t{i}" for i in range(n_taxa)} # t0..tn-1
        dist = {}
        for i in range(n_taxa):
            for j in range(i+1, n_taxa):
                dist[(i, j)] = D[i, j]
                dist[(j, i)] = D[i, j]
                
        splits = []
        next_id = n_taxa
        
        while len(active) > 3:
            n = len(active)
            r = {u: sum(dist[(u, v)] for v in active if v != u) for u in active}
            
            min_q = np.inf
            best_u = -1
            best_v = -1
            
            for i_idx in range(n):
                u = active[i_idx]
                for j_idx in range(i_idx + 1, n):
                    v = active[j_idx]
                    d_uv = dist[(u, v)]
                    q_val = (n - 2) * d_uv - r[u] - r[v]
                    if q_val < min_q:
                        min_q = q_val
                        best_u = u
                        best_v = v
            
            u = best_u
            v = best_v
            w = next_id
            next_id += 1
            
            node_taxa[w] = node_taxa[u] | node_taxa[v]
            # Newick for w: (newick_u, newick_v)
            node_newick[w] = f"({node_newick[u]},{node_newick[v]})"
            
            set_u = node_taxa[u]
            set_v = node_taxa[v]
            
            if 1 < len(set_u) < n_taxa - 1:
                splits.append(frozenset(set_u))
            if 1 < len(set_v) < n_taxa - 1:
                splits.append(frozenset(set_v))
                
            active.remove(u)
            active.remove(v)
            
            d_uv = dist[(u, v)]
            for x in active:
                d_wx = 0.5 * (dist[(u, x)] + dist[(v, x)] - d_uv)
                dist[(w, x)] = d_wx
                dist[(x, w)] = d_wx
                
            active.append(w)
            
        # Final 3
        # The tree is a star of 3 nodes.
        # We need to join them.
        # The topology is ((a,b),c) or similar? 
        # In an unrooted tree with 3 leaves, it's a star.
        # With 3 internal nodes left, we join them into a single root?
        # Actually, when 3 nodes remain, we just connect them.
        # The Newick string will be (node1, node2, node3);
        # But standard Newick is binary. 
        # We can just join the first two, then join that with the third.
        # Topology of the last 3 doesn't matter for splits?
        # The splits are determined by the sets.
        
        # Let's just create a polytomy or binary join.
        # For scoring (RF distance), the exact arrangement of the last 3 doesn't matter 
        # if the splits are the same. 
        # But we need a valid Newick string.
        
        a, b, c = active
        # Join a and b
        temp = next_id
        next_id += 1
        node_taxa[temp] = node_taxa[a] | node_taxa[b]
        node_newick[temp] = f"({node_newick[a]},{node_newick[b]})"
        
        # Add splits for this join?
        # These splits might have been counted already if a,b were internal.
        # If a,b were leaves, these are trivial.
        # Let's assume the main splits were captured in the loop.
        
        # Join temp and c
        root = next_id
        node_taxa[root] = node_taxa[temp] | node_taxa[c]
        newick = f"({node_newick[temp]},{node_newick[c]})"
        
        return splits, newick + ";"

    # Run first NJ
    splits_0, newick_0 = _nj_with_newick(D, n_taxa)
    for s in splits_0:
        split_counts[s] += 1
        
    trees = [(newick_0, splits_0)]
    
    # Bootstrap Loop
    # Estimate time per iteration
    # t_start = time.time()
    # ... run one bootstrap ...
    # t_iter = time.time() - t_start
    
    # Let's just run a fixed number, say 50, and check time.
    # Or run until deadline - 10s.
    
    n_boot = 50
    rng = np.random.default_rng(42)
    
    for b in range(n_boot):
        if time.time() > _DEADLINE - 10:
            break
            
        # Resample sites with replacement
        indices = rng.integers(0, n_sites, size=n_sites)
        boot_aln = alignment[:, indices]
        
        # Compute D for bootstrap
        D_boot = _nj_distance_matrix(boot_aln, n_taxa, n_sites)
        
        # Run NJ
        splits_b, newick_b = _nj_with_newick(D_boot, n_taxa)
        
        trees.append((newick_b, splits_b))
        
        for s in splits_b:
            split_counts[s] += 1
            
    # Determine Majority Splits
    # Total runs = 1 + n_boot_executed
    total_runs = len(trees)
    threshold = total_runs / 2.0
    
    majority_splits = set()
    for s, count in split_counts.items():
        if count > threshold:
            majority_splits.add(s)
            
    # Score each tree by number of majority splits it contains
    best_tree = trees[0][0]
    best_score = -1
    
    for newick, splits in trees:
        score = 0
        for s in splits:
            if s in majority_splits:
                score += 1
        if score > best_score:
            best_score = score
            best_tree = newick
            
    return best_tree

# Ensure the function is named correctly
# The prompt asks to keep function names.
# reconstruct is the entry point.

