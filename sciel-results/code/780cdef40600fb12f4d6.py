"""Find a graph on up to 22 vertices with no regular induced subgraph of size >= 6."""
import time
import random
import itertools

# Deadline guard: evaluator kills after 150s, leave safety margin
DEADLINE = time.time() + 118  # 120s limit, 2s safety
SAFETY = 2.0


def _has_violation_full(adj, n):
    """Check if any induced subgraph of size >= 6 is regular.
    Returns (True, subset) or (False, None).
    """
    for k in range(6, n + 1):
        for subset in itertools.combinations(range(n), k):
            mask = 0
            for v in subset:
                mask |= (1 << v)
            
            degs = []
            for v in subset:
                # Count neighbors of v within the subset
                # adj[v] is a bitmask of neighbors. 
                # Intersection with mask gives neighbors in subset.
                # bit_count() gives the degree.
                d = (adj[v] & mask).bit_count()
                degs.append(d)
            
            if all(d == degs[0] for d in degs):
                return True, subset
    return False, None


def _count_v6_violations(adj, n):
    """Count regular induced subgraphs of size exactly 6.
    Returns (count, list_of_violating_subsets).
    Early exits if count > 50.
    """
    count = 0
    violations = []
    for subset in itertools.combinations(range(n), 6):
        mask = 0
        for v in subset:
            mask |= (1 << v)
        
        degs = []
        for v in subset:
            d = (adj[v] & mask).bit_count()
            degs.append(d)
        
        if all(d == degs[0] for d in degs):
            count += 1
            if len(violations) < 10:
                violations.append(subset)
            if count > 50:
                return count, violations
    return count, violations


def _local_search_v6(adj, n, deadline):
    """Local search to minimize size-6 violations.
    Returns (adj, v6_count, full_violation_subset).
    """
    current_count, current_subs = _count_v6_violations(adj, n)
    
    # If no size-6 violations, check for larger ones
    if current_count == 0:
        has_v, subset = _has_violation_full(adj, n)
        if not has_v:
            return adj, 0, None
        else:
            # We have a violation of size > 6
            return adj, 0, subset

    best_adj = adj[:]
    best_count = current_count
    best_full_viol = None
    
    # If we start with 0 V6 violations but have a larger one, we handle that in the main loop
    # Here we assume current_count > 0
    
    for _ in range(10000):  # Max iterations
        if time.time() > deadline:
            break
        
        if current_count == 0:
            # Check full
            has_v, subset = _has_violation_full(adj, n)
            if not has_v:
                return adj, 0, None
            else:
                # Try to fix the larger violation
                # Flip edges within this subset
                fixed = False
                for i in range(len(subset)):
                    for j in range(i + 1, len(subset)):
                        u, v = subset[i], subset[j]
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                        
                        # Check if this fixed the full violation
                        new_has_v, _ = _has_violation_full(adj, n)
                        if not new_has_v:
                            return adj, 0, None
                        
                        # Revert
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                
                # If we couldn't fix it with single flips, try random restart
                # Flip a random edge
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                
                # Update counts
                current_count, current_subs = _count_v6_violations(adj, n)
                if current_count > 0:
                    # We introduced V6 violations, continue search
                    pass
                else:
                    # Still no V6 violations, loop will check full again
                    pass
                continue
        
        # Normal case: current_count > 0
        # Pick a violating subset
        subset = current_subs[0]
        improved = False
        
        # Try flipping edges within the subset
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                u, v = subset[i], subset[j]
                adj[u] ^= (1 << v)
                adj[v] ^= (1 << u)
                
                new_count, _ = _count_v6_violations(adj, n)
                
                if new_count < best_count:
                    best_count = new_count
                    best_adj = adj[:]
                    improved = True
                    if new_count == 0:
                        # Check full
                        has_v, subset_v = _has_violation_full(adj, n)
                        if not has_v:
                            return adj, 0, None
                        else:
                            # Save state and break to handle larger violation
                            current_count = 0
                            current_subs = []
                            break
                else:
                    # Revert
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                
                if time.time() > deadline:
                    return best_adj, best_count, None
        
        if not improved:
            # Try flipping edges incident to subset vertices
            for u in subset:
                for v in range(n):
                    if v in subset:
                        continue
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
                    
                    new_count, _ = _count_v6_violations(adj, n)
                    
                    if new_count < best_count:
                        best_count = new_count
                        best_adj = adj[:]
                        improved = True
                        if new_count == 0:
                            has_v, subset_v = _has_violation_full(adj, n)
                            if not has_v:
                                return adj, 0, None
                            else:
                                current_count = 0
                                current_subs = []
                                break
                    else:
                        adj[u] ^= (1 << v)
                        adj[v] ^= (1 << u)
                    
                    if time.time() > deadline:
                        return best_adj, best_count, None
                    if current_count == 0:
                        break
        
        if not improved:
            # Random restart
            for _ in range(5):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u != v:
                    adj[u] ^= (1 << v)
                    adj[v] ^= (1 << u)
            
            current_count, current_subs = _count_v6_violations(adj, n)
            if current_count < best_count:
                best_count = current_count
                best_adj = adj[:]
        
        # Update current state for next iteration
        # We only kept best_adj if improved, but we modified adj in place.
        # If we reverted, adj is back to previous state.
        # If we improved, adj is the new state.
        # So current_count should be recalculated if we want to be safe, 
        # but we already calculated new_count for the improved step.
        # If we didn't improve, adj is unchanged, so current_count is still valid.
        
        # However, if we broke out of the inner loop due to time or count==0, 
        # we need to handle that.
        # The loop structure above handles the 'break' by setting current_count=0.
        
    return best_adj, best_count, None


def construct():
    """Construct a graph with no regular induced subgraph of size >= 6.
    Maximize n.
    """
    start_time = time.time()
    
    best_n = 5
    best_edges = []
    
    # Try n=21 first, then 20
    # Allocate time: 60s for 21, 50s for 20
    time_for_21 = start_time + 60
    time_for_20 = start_time + 115
    
    for n in [21, 20, 19, 18]:
        deadline = time_for_21 if n == 21 else time_for_20
        if time.time() > deadline - 2:
            continue
        
        found = False
        max_restarts = 20
        
        for restart in range(max_restarts):
            if time.time() > deadline - 2:
                break
            
            # Initialize with random graph
            adj = [0] * n
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < 0.5:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Local search
            adj, final_v6_count, full_viol = _local_search_v6(adj, n, deadline)
            
            if final_v6_count == 0 and full_viol is None:
                # Verify one last time
                has_v, _ = _has_violation_full(adj, n)
                if not has_v:
                    found = True
                    best_n = n
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj[u] >> v) & 1:
                                edges.append([u, v])
                    best_edges = edges
                    break
        
        if found:
            # If we found 21, we can stop. If we found 20, we can try 21? 
            # No, we tried 21 first. 
            # If we found 20, we can return it.
            break
    
    # If we didn't find anything >= 18, try smaller
    if best_n < 18:
        for n in range(17, 5, -1):
            if time.time() > DEADLINE - 5:
                break
            # Simple random search for smaller n
            for _ in range(100):
                if time.time() > DEADLINE - 5:
                    break
                adj = [0] * n
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < 0.5:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                has_v, _ = _has_violation_full(adj, n)
                if not has_v:
                    best_n = n
                    edges = []
                    for u in range(n):
                        for v in range(u + 1, n):
                            if (adj[u] >> v) & 1:
                                edges.append([u, v])
                    best_edges = edges
                    break
            if best_n == n:
                break
    
    return {'n': best_n, 'edges': best_edges}
