"""Rewritten program for the suite task; Hybrid MIS + Local Search for sum-free subsets."""
# EVOLVE-BLOCK-START
import time
from random import Random

def _instance(size, seed, limit=10000):
    rng = Random(seed)
    values = set()
    while len(values) < size:
        x = rng.randint(-limit, limit)
        if x != 0:
            values.add(x)
    return sorted(values)

def _is_sum_free(S):
    s = set(S)
    for a in S:
        for b in S:
            if (a + b) in s:
                return False
    return True

def _can_add(S_set, x):
    """Check if x can be added to S_set without violating sum-freeness."""
    if 2 * x in S_set:
        return False
    for c in S_set:
        if x + c in S_set:
            return False
    # Check if x can be formed as sum of two elements in S_set
    for c1 in S_set:
        if x - c1 in S_set:
            return False
    return True

def _greedy_by_key(S, key_func):
    """Greedy construction: add elements in order of key_func if they don't conflict."""
    S_set = set()
    result = []
    for x in sorted(S, key=key_func):
        if _can_add(S_set, x):
            S_set.add(x)
            result.append(x)
    return result

def _greedy_odd(S):
    return [x for x in S if x % 2 != 0]

def _local_search(S, rng, deadline):
    """Improve a sum-free subset of S via local search with efficient conflict checking."""
    S_set = set(S)
    all_elements = set(S)
    excluded = list(all_elements - S_set)
    
    best = len(S_set)
    best_set = set(S_set)
    
    # Ensure initial is valid
    if not _is_sum_free(S_set):
        S_set = set(_greedy_by_key(list(S), lambda v: -abs(v)))
        excluded = list(all_elements - S_set)
        best = len(S_set)
        best_set = set(S_set)
    
    while time.time() < deadline:
        improved = False
        
        # Try adding each excluded element
        for x in list(excluded):
            if time.time() >= deadline:
                break
            if _can_add(S_set, x):
                S_set.add(x)
                excluded.remove(x)
                improved = True
                if len(S_set) > best:
                    best = len(S_set)
                    best_set = set(S_set)
        
        # Try swaps: add x, remove one conflicting member
        for x in list(excluded):
            if time.time() >= deadline:
                break
            if _can_add(S_set, x):
                continue
            # Find conflicts
            conflicts = set()
            if 2 * x in S_set:
                conflicts.add(2 * x)
            for c in S_set:
                if x + c in S_set:
                    conflicts.add(c)
                    conflicts.add(x + c)
            for c1 in S_set:
                if x - c1 in S_set:
                    conflicts.add(c1)
                    conflicts.add(x - c1)
            
            for c in conflicts:
                if c in S_set:
                    # Check if x is compatible with S_set - {c}
                    if _can_add(S_set - {c}, x):
                        S_set.discard(c)
                        S_set.add(x)
                        excluded.remove(x)
                        improved = True
                        if len(S_set) > best:
                            best = len(S_set)
                            best_set = set(S_set)
                        break
        
        # Try 2-for-1: add two excluded, remove one member
        if len(excluded) >= 2:
            for i in range(len(excluded)):
                if time.time() >= deadline:
                    break
                x1 = excluded[i]
                for j in range(i + 1, len(excluded)):
                    x2 = excluded[j]
                    for c in list(S_set):
                        if _can_add(S_set - {c}, x1):
                            temp_set = (S_set - {c}) | {x1}
                            if _can_add(temp_set, x2):
                                S_set.discard(c)
                                S_set.add(x1)
                                S_set.add(x2)
                                excluded.remove(x1)
                                excluded.remove(x2)
                                improved = True
                                if len(S_set) > best:
                                    best = len(S_set)
                                    best_set = set(S_set)
                                break
                    if improved:
                        break
                if improved:
                    break
        
        if not improved:
            rng.shuffle(excluded)
            any_move = False
            for x in list(excluded):
                if time.time() >= deadline:
                    break
                if _can_add(S_set, x):
                    S_set.add(x)
                    excluded.remove(x)
                    any_move = True
                    if len(S_set) > best:
                        best = len(S_set)
                        best_set = set(S_set)
            if not any_move:
                break
    
    return list(best_set)

def _solve_mis(S, deadline):
    """
    Finds a large Independent Set in the conflict graph G' where
    E' = {(u,v) : u != v, u+v in S} U {(u, 2u) : 2u in S}.
    Uses Branch and Bound with bitsets.
    Returns the set of vertices in the MIS found.
    """
    n = len(S)
    val_to_idx = {v: i for i, v in enumerate(S)}
    
    # Build adjacency bitsets
    # adj[i] is a bitmask of neighbors of vertex i
    adj = [0] * n
    
    for i in range(n):
        u = S[i]
        for j in range(n):
            if i == j: continue
            v = S[j]
            s_val = u + v
            if s_val in val_to_idx:
                # Edge (i, j) exists if u+v is in S
                # Note: This edge prevents u and v from being in the independent set
                # because if u,v in IS, then u+v in S, but for IS to be sum-free,
                # we need u+v NOT in IS. 
                # Wait, the logic derived earlier:
                # If (u,v) in E', then u and v cannot both be in the Independent Set.
                # Why? Because if u,v in IS, and u+v in S, it doesn't violate IS property directly.
                # It violates Sum-Free property if u+v is ALSO in IS.
                # My previous derivation:
                # "If u,v in I and u+v in S, then (u,v) in E'. Since I is independent, we cannot have both u,v in I."
                # This implies that the definition of E' must be such that the edge represents a direct conflict for the Independent Set property.
                # But an edge in a graph for MIS just means "cannot pick both".
                # So, if we define E' such that (u,v) in E' whenever their co-existence violates sum-freeness?
                # No, co-existence of u and v only violates sum-freeness if u+v is also in the set.
                # This is a 3-way constraint.
                
                # Let's re-evaluate the MIS mapping.
                # If we pick an Independent Set I in G', is it sum-free?
                # If I is independent, then for any u,v in I, (u,v) is NOT an edge.
                # If we define (u,v) as an edge if u+v in S, then u,v in I implies u+v NOT in S.
                # If u+v NOT in S, then u+v cannot be in I (since I subset S).
                # So u+v not in I.
                # This satisfies the sum-free condition for distinct u,v.
                # What about u=v?
                # If 2u in S, we need 2u not in I.
                # If we add edge (u, 2u), then u and 2u cannot both be in I.
                # So if u in I, 2u not in I.
                # This works!
                
                # So the graph G' is:
                # Vertices: S
                # Edges:
                # 1. (u, v) if u != v and u+v in S.
                # 2. (u, 2u) if 2u in S.
                
                # This graph's Independent Sets are Sum-Free.
                
                # Is the Max Sum-Free set necessarily an Independent Set in G'?
                # No. As shown before, if u+v in S but u+v not in I, then u,v can be in I (Sum-Free) but (u,v) is an edge in G' (so not Independent).
                # So MIS(G') is a subset of Sum-Free sets.
                # It might be smaller.
                
                # However, for sparse random graphs, this is a good heuristic.
                
                pass # Placeholder, logic is in the loop below

    # Re-build adj correctly
    for i in range(n):
        u = S[i]
        mask = 0
        for j in range(n):
            if i == j: continue
            v = S[j]
            if u + v in val_to_idx:
                mask |= (1 << j)
        # Handle self-sum 2u
        if 2 * u in val_to_idx:
            idx_2u = val_to_idx[2 * u]
            mask |= (1 << idx_2u)
        adj[i] = mask

    # Branch and Bound
    # State: available vertices mask, current set size, current set list
    # To save memory and time, we just track the mask and size.
    # We need to reconstruct the set at the end, so we store the path or just return the mask.
    
    best_mask = 0
    best_size = 0
    
    # Heuristic: Order vertices by degree? 
    # For B&B, we pick a vertex from the available mask.
    
    def rec(mask, depth, current_size, current_mask):
        nonlocal best_mask, best_size
        if time.time() > deadline:
            return
        if mask == 0:
            if current_size > best_size:
                best_size = current_size
                best_mask = current_mask
            return
        
        # Bound: if current_size + popcount(mask) <= best_size, prune
        # Python int bit_count is fast
        if current_size + mask.bit_count() <= best_size:
            return
            
        # Pick a vertex
        # Heuristic: pick vertex with max degree in the induced subgraph?
        # Or just the first one?
        # Let's pick the one with the highest index for simplicity, or max degree.
        # Max degree calculation is expensive.
        # Let's try picking the vertex that removes the most neighbors (max degree in induced subgraph).
        
        # Find a vertex v in mask
        # We can iterate through bits
        # To optimize, we can pre-compute something?
        # For now, pick the first available vertex.
        # A better heuristic: pick v such that adj[v] & mask has max popcount.
        
        # Let's try a simple heuristic: pick v with max degree in induced subgraph.
        # This requires iterating all v in mask.
        
        v_idx = -1
        max_deg = -1
        temp_mask = mask
        while temp_mask:
            lsb = temp_mask & -temp_mask
            v = lsb.bit_length() - 1
            deg = (adj[v] & mask).bit_count()
            if deg > max_deg:
                max_deg = deg
                v_idx = v
            temp_mask ^= lsb
            
        if v_idx == -1:
            return
            
        v_bit = 1 << v_idx
        neighbors_mask = adj[v_idx] & mask
        
        # Branch 1: Include v
        # New mask = mask - v - neighbors
        new_mask_include = mask & ~v_bit & ~neighbors_mask
        rec(new_mask_include, depth + 1, current_size + 1, current_mask | v_bit)
        
        # Branch 2: Exclude v
        new_mask_exclude = mask & ~v_bit
        rec(new_mask_exclude, depth + 1, current_size, current_mask)

    full_mask = (1 << n) - 1
    rec(full_mask, 0, 0, 0)
    
    # Reconstruct set from best_mask
    result = []
    for i in range(n):
        if best_mask & (1 << i):
            result.append(S[i])
            
    return result

def construct():
    deadline = time.time() + 145.0
    results = []
    instances = [(60, 801), (120, 802), (240, 803)]
    
    for idx, (size, seed) in enumerate(instances):
        S = _instance(size, seed)
        rng = Random(seed + 12345)
        
        remaining_time = deadline - time.time()
        num_remaining = len(instances) - idx
        inst_budget = remaining_time / num_remaining - 2.0
        inst_deadline = time.time() + inst_budget
        
        # Build multiple initial candidates
        cand1 = _greedy_odd(S)
        cand2 = _greedy_by_key(S, lambda v: -abs(v))
        cand3 = _greedy_by_key(S, lambda v: v)
        cand4 = _greedy_by_key(S, lambda v: -v)
        
        # Try MIS solver
        # Allocate 70% of time to MIS, 30% to Local Search
        mis_deadline = time.time() + inst_budget * 0.7
        try:
            cand5 = _solve_mis(S, mis_deadline)
            if not _is_sum_free(cand5):
                cand5 = [] # Fallback if invalid
        except Exception:
            cand5 = []
            
        candidates = [cand1, cand2, cand3, cand4, cand5]
        best_init = max(candidates, key=len)
        
        # Local Search with remaining time
        ls_deadline = time.time() + inst_budget * 0.3
        # Ensure we don't exceed inst_deadline
        ls_deadline = min(ls_deadline, inst_deadline)
        
        result = _local_search(best_init, rng, ls_deadline)
        
        if not _is_sum_free(result):
            result = best_init
            if not _is_sum_free(result):
                result = cand1
        
        results.append(result)
    
    return results
# EVOLVE-BLOCK-END
