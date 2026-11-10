"""Find a graph on up to 22 vertices with no regular induced subgraph of size >= 6."""
import time
import random
import itertools
from collections import defaultdict

# Deadline guard: evaluator kills after 150s, leave safety margin
DEADLINE = time.time() + 120  # 120s wall-clock limit, use 115s for safety
SAFETY_MARGIN = 5  # seconds before limit to stop

def _is_regular_6(mask, adj):
    """Check if the 6-subset defined by bitmask 'mask' is regular in graph 'adj'.
    adj is a list of bitmasks.
    """
    # Extract vertices
    # Since mask has exactly 6 bits set, we can iterate
    degs = []
    m = mask
    while m:
        # Get lowest set bit
        lsb = m & -m
        v = lsb.bit_length() - 1
        m ^= lsb
        # Count neighbors of v within mask
        degs.append((adj[v] & mask).bit_count())
    
    # Check if all degrees are equal
    first = degs[0]
    for d in degs[1:]:
        if d != first:
            return False
    return True

def _count_violations_all(adj, n):
    """Count all regular induced subgraphs of size >= 6.
    Returns (total_count, list_of_violating_subsets_masks)
    """
    count = 0
    violations = []
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            # Check regularity
            degs = []
            m = mask
            while m:
                lsb = m & -m
                v = lsb.bit_length() - 1
                m ^= lsb
                degs.append((adj[v] & mask).bit_count())
            
            if all(d == degs[0] for d in degs):
                count += 1
                if len(violations) < 20:
                    violations.append(mask)
                if count > 50:
                    return count, violations
    return count, violations

def _local_search_fast(adj, n):
    """Local search with incremental evaluation for size-6 subsets.
    Returns (adj, violations_count_all)
    """
    # Precompute all 6-subsets
    subsets_6 = []
    for subset in itertools.combinations(range(n), 6):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        subsets_6.append(mask)
    
    # Map each edge to the indices of 6-subsets it belongs to
    # Edge is (u, v) with u < v
    edge_to_subsets = defaultdict(list)
    for idx, mask in enumerate(subsets_6):
        # Extract vertices from mask
        verts = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            verts.append(v)
        # For each pair in verts, add idx to edge_to_subsets
        for i in range(6):
            for j in range(i + 1, 6):
                u, v = verts[i], verts[j]
                if u > v: u, v = v, u
                edge_to_subsets[(u, v)].append(idx)
    
    # Initial count
    current_count_6 = sum(1 for m in subsets_6 if _is_regular_6(m, adj))
    
    if current_count_6 == 0:
        # Check larger sizes
        total_count, _ = _count_violations_all(adj, n)
        if total_count == 0:
            return adj, 0
        # If size 6 is clean but larger are not, we need to handle that.
        # For now, return as is if we can't improve size 6 further?
        # Or continue searching to break larger structures?
        # Let's try to break larger structures by random flips if stuck at 0 size-6 violations.
    
    best_adj = adj[:]
    best_count_6 = current_count_6
    
    # Local search loop
    max_iters = 50000
    for it in range(max_iters):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        if current_count_6 == 0:
            # Check if we have a valid graph
            total_count, _ = _count_violations_all(adj, n)
            if total_count == 0:
                return adj, 0
            
            # If we have size > 6 violations, try to break them
            # Pick a random edge and flip it, see if total violations decrease
            # This is expensive, so do it sparingly
            if it % 100 == 0:
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    new_total, _ = _count_violations_all(adj, n)
                    if new_total < total_count:
                        # Keep flip
                        # Update current_count_6
                        current_count_6 = sum(1 for m in subsets_6 if _is_regular_6(m, adj))
                        total_count = new_total
                        best_adj = adj[:]
                        best_count_6 = current_count_6
                    else:
                        # Revert
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
            continue
        
        # Pick a violating 6-subset to fix
        # Find one
        viol_idx = -1
        for idx, m in enumerate(subsets_6):
            if _is_regular_6(m, adj):
                viol_idx = idx
                break
        
        if viol_idx == -1:
            # Should not happen if current_count_6 > 0
            break
        
        # Pick a random edge within this violating subset
        verts = []
        m = subsets_6[viol_idx]
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            verts.append(v)
        
        i = random.randint(0, 5)
        j = random.randint(0, 5)
        if i == j:
            j = (j + 1) % 6
        u, v = verts[i], verts[j]
        if u > v: u, v = v, u
        
        # Flip edge (u, v)
        adj[u] ^= (1 << v)
        adj[v] ^= (1 << u)
        
        # Update count incrementally
        # Only subsets containing (u, v) are affected
        affected_indices = edge_to_subsets.get((u, v), [])
        
        # Calculate change in count
        delta = 0
        for idx in affected_indices:
            mask = subsets_6[idx]
            was_regular = _is_regular_6(mask, adj) # adj is already flipped
            # We need to know if it was regular BEFORE flip
            # Let's revert, check, then flip again? No, that's slow.
            # Better: Check current state.
            # If current is regular, it contributes +1.
            # We need the previous state.
            # Let's just recalculate the count for the affected subsets.
            pass
        
        # Simpler: Just recalculate current_count_6 fully if the delta logic is complex?
        # No, incremental is the point.
        # Let's do this:
        # Save old regularity of affected subsets
        old_regular = [False] * len(affected_indices)
        for k, idx in enumerate(affected_indices):
             # Revert temporarily to check old state?
             # Or just check new state and compare to a stored "regularity" array?
             # Let's maintain an array `is_reg` of length len(subsets_6)
             pass
        
        # This is getting complicated. Let's use a simpler heuristic:
        # Flip, then check if the specific violating subset is now non-regular.
        # If yes, keep flip. If no, revert.
        # This is a "greedy" move. It might not reduce total count, but it fixes one.
        
        is_now_regular = _is_regular_6(subsets_6[viol_idx], adj)
        if not is_now_regular:
            # We fixed this one. But did we break others?
            # Check total count_6
            new_count_6 = sum(1 for m in subsets_6 if _is_regular_6(m, adj))
            if new_count_6 < current_count_6:
                current_count_6 = new_count_6
                if current_count_6 < best_count_6:
                    best_count_6 = current_count_6
                    best_adj = adj[:]
            else:
                # Revert
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
        else:
            # Revert
            adj[u] ^= (1 << v)
            adj[v] ^= (1 << u)
            
        # Random restart if stuck
        if it % 500 == 0 and current_count_6 == best_count_6:
             # Flip random edges
             for _ in range(5):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
             current_count_6 = sum(1 for m in subsets_6 if _is_regular_6(m, adj))
             if current_count_6 < best_count_6:
                 best_count_6 = current_count_6
                 best_adj = adj[:]
                 
    # Final check
    total_count, _ = _count_violations_all(best_adj, n)
    if total_count == 0:
        return best_adj, 0
    
    return best_adj, total_count

def construct():
    """Construct a graph with no regular induced subgraph of size >= 6.
    Maximize n.
    """
    start_time = time.time()
    
    best_n = 5
    best_edges = []
    
    # Try to build graphs of increasing size
    # Start with n=10 and work up
    
    for n in range(10, 23):
        if time.time() > DEADLINE - 10:
            break
        
        found = False
        max_restarts = 20 if n <= 15 else 10
        
        for restart in range(max_restarts):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            # Initialize with random graph
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Local search
            adj, final_violations = _local_search_fast(adj, n)
            
            if final_violations == 0:
                found = True
                break
        
        if found:
            best_n = n
            edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if (adj[u] >> v) & 1:
                        edges.append([u, v])
            best_edges = edges
        else:
            continue
    
    # Verify the result
    if best_n >= 6 and best_edges:
        adj = [0] * best_n
        for u, v in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        violations, _ = _count_violations_all(adj, best_n)
        if violations != 0:
            best_n = 5
            best_edges = []
    
    return {'n': best_n, 'edges': best_edges}
