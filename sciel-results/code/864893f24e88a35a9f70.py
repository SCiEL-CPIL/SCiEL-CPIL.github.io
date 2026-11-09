"""Four-cycle-free edge sets of Q7: maximize edges with no 2-face fully selected."""
import time
import random

# EVOLVE-BLOCK-START

def _greedy_deletion(all_c4s, c4_membership, num_c4s, total_edges, deadline, seed):
    """Run greedy deletion with a given random seed for tie-breaking."""
    rng = random.Random(seed)
    
    present = [True] * total_edges
    c4_count = [6] * total_edges  # Each edge is in exactly 6 C4s in Q7
    
    bad_c4s = set(range(num_c4s))
    
    while bad_c4s:
        if time.time() > deadline - 20:
            break
        
        max_count = -1
        candidates = []
        for e in range(total_edges):
            if present[e] and c4_count[e] > max_count:
                max_count = c4_count[e]
                candidates = [e]
            elif present[e] and c4_count[e] == max_count:
                candidates.append(e)
        
        if not candidates:
            break
        
        # Pick randomly among tied candidates
        best_e = candidates[rng.randint(0, len(candidates) - 1)]
        
        # Delete best_e
        present[best_e] = False
        
        # Update counts for C4s containing best_e
        for c4_idx in c4_membership[best_e]:
            if c4_idx in bad_c4s:
                bad_c4s.remove(c4_idx)
                e1, e2, e3, e4 = all_c4s[c4_idx]
                for other_e in (e1, e2, e3, e4):
                    if other_e != best_e:
                        c4_count[other_e] -= 1
    
    return present

def _try_add_edge(present, e, all_c4s, c4_membership):
    """Check if edge e can be added without creating a bad C4."""
    for c4_idx in c4_membership[e]:
        e1, e2, e3, e4 = all_c4s[c4_idx]
        others = [x for x in (e1, e2, e3, e4) if x != e]
        if all(present[x] for x in others):
            return False
    return True

def _local_search(present, all_c4s, c4_membership, total_edges, deadline, seed):
    """Local search: try to add removed edges, then try swap moves."""
    rng = random.Random(seed)
    m = sum(present)
    
    # Phase 1: Try to add back removed edges
    removed = [e for e in range(total_edges) if not present[e]]
    rng.shuffle(removed)
    
    for _pass in range(20):
        if time.time() > deadline - 5:
            break
        progress = False
        for e in removed:
            if present[e]:
                continue
            if _try_add_edge(present, e, all_c4s, c4_membership):
                present[e] = True
                m += 1
                progress = True
        if not progress:
            break
    
    # Phase 2: Swap moves - remove one edge, try to add two others
    # For each removed edge that is blocking others, see if we can do a beneficial swap
    for _iter in range(50):
        if time.time() > deadline - 5:
            break
        
        # Try swap: for each present edge, remove it and see if we can add more than 1 back
        # Focus on edges that are in many C4s with other present edges
        best_swap_gain = 0
        best_swap = None
        
        # Sample a subset of present edges to try (for speed)
        present_edges = [e for e in range(total_edges) if present[e]]
        rng.shuffle(present_edges)
        
        for e_remove in present_edges[:50]:  # Try at most 50 candidates
            # Temporarily remove e_remove
            present[e_remove] = False
            
            # Count how many removed edges can now be added
            addable = []
            for e_add in removed:
                if present[e_add]:
                    continue
                if _try_add_edge(present, e_add, all_c4s, c4_membership):
                    addable.append(e_add)
            
            gain = len(addable) - 1  # Net gain (remove 1, add len(addable))
            
            if gain > best_swap_gain:
                best_swap_gain = gain
                best_swap = (e_remove, addable)
            
            # Restore
            present[e_remove] = True
        
        if best_swap is None or best_swap_gain <= 0:
            break
        
        e_remove, addable = best_swap
        present[e_remove] = False
        m -= 1
        for e_add in addable:
            present[e_add] = True
            m += 1
        
        # Update removed list
        removed = [e for e in range(total_edges) if not present[e]]
    
    return m

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    d = 7
    n = 1 << d  # 128
    total_edges = n * d // 2  # 448

    # 1. Generate all edges and map to indices
    all_edges = []
    edge_index = {}
    for x in range(n):
        for i in range(d):
            y = x ^ (1 << i)
            if x < y:
                idx = len(all_edges)
                all_edges.append((x, y))
                edge_index[(x, y)] = idx

    # 2. Precompute edge info (u, v, bit)
    edge_info = []
    for idx, (u, v) in enumerate(all_edges):
        diff = u ^ v
        bit = diff.bit_length() - 1
        edge_info.append((u, v, bit))

    # 3. Precompute all C4s (faces)
    all_c4s = []
    for i in range(d):
        for j in range(i + 1, d):
            mask = (1 << i) | (1 << j)
            for x in range(n):
                if (x & mask) == 0:
                    v1 = x
                    v2 = x ^ (1 << i)
                    v3 = x ^ (1 << j)
                    v4 = x ^ (1 << i) ^ (1 << j)
                    
                    e1 = edge_index[(min(v1, v2), max(v1, v2))]
                    e2 = edge_index[(min(v3, v4), max(v3, v4))]
                    e3 = edge_index[(min(v1, v3), max(v1, v3))]
                    e4 = edge_index[(min(v2, v4), max(v2, v4))]
                    
                    all_c4s.append((e1, e2, e3, e4))

    num_c4s = len(all_c4s)
    
    # 4. Precompute for each edge, the list of C4 indices it belongs to
    c4_membership = [[] for _ in range(total_edges)]
    for c4_idx, (e1, e2, e3, e4) in enumerate(all_c4s):
        for e in (e1, e2, e3, e4):
            c4_membership[e].append(c4_idx)

    # 5. Multiple random restarts of greedy deletion + local search
    best_present = None
    best_m = 0
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        seed = 42 + restart * 1000
        present = _greedy_deletion(all_c4s, c4_membership, num_c4s, total_edges, deadline, seed)
        m = _local_search(present, all_c4s, c4_membership, total_edges, deadline, seed)
        
        if m > best_m:
            best_m = m
            best_present = present[:]
    
    # 6. Final local search on the best solution
    if best_present is not None:
        final_m = _local_search(best_present, all_c4s, c4_membership, total_edges, deadline, 99999)
    
    # 7. Construct Output
    out = []
    for idx, (u, v) in enumerate(all_edges):
        if best_present[idx]:
            out.append([u, v])
            
    return out

# EVOLVE-BLOCK-END
