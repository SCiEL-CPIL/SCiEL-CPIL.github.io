import time
import random
import itertools
import sys

DEADLINE = 150
SAFETY_MARGIN = 10
START_TIME = None

def check_size6_violations_containing(adj_masks, n, u, v):
    """
    Checks if adding edge (u, v) to the graph represented by adj_masks
    creates a regular induced subgraph of size 6 containing u and v.
    Assumes edge (u, v) is NOT currently present.
    """
    # We need to check all subsets S of size 6 such that u, v in S.
    # The other 4 vertices are chosen from V \ {u, v}.
    others = [x for x in range(n) if x != u and x != v]
    
    # Precompute current degrees of all vertices? 
    # Or just compute for the subset.
    
    for combo_others in itertools.combinations(others, 4):
        S = (u, v) + tuple(combo_others)
        mask = 0
        for x in S:
            mask |= (1 << x)
        
        # Compute degrees in current graph (without u-v edge)
        # deg(x) = (adj_masks[x] & mask).bit_count()
        # Note: u and v are not connected yet in adj_masks
        
        du = (adj_masks[u] & mask).bit_count()
        dv = (adj_masks[v] & mask).bit_count()
        
        # If we add edge (u, v), new degrees are:
        new_du = du + 1
        new_dv = dv + 1
        
        # Check if all vertices in S have the same new degree
        target_deg = new_du
        if new_dv != target_deg:
            continue
            
        is_regular = True
        for x in S[2:]:
            if (adj_masks[x] & mask).bit_count() != target_deg:
                is_regular = False
                break
        
        if is_regular:
            return True
            
    return False

def find_violation(adj_masks, n, min_size=6):
    """
    Finds any regular induced subgraph of size >= min_size.
    Returns the tuple of vertices or None.
    """
    # Check size 6 first
    for combo in itertools.combinations(range(n), 6):
        mask = 0
        for v in combo:
            mask |= (1 << v)
        
        d0 = (adj_masks[combo[0]] & mask).bit_count()
        is_regular = True
        for v in combo[1:]:
            if (adj_masks[v] & mask).bit_count() != d0:
                is_regular = False
                break
        if is_regular:
            return combo
            
    # Check size 7
    if n >= 7:
        for combo in itertools.combinations(range(n), 7):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            is_regular = True
            for v in combo[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_regular = False
                    break
            if is_regular:
                return combo
                
    # Check size 8+ (unlikely to be found quickly if 6,7 are clean)
    for k in range(8, n + 1):
        for combo in itertools.combinations(range(n), k):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            d0 = (adj_masks[combo[0]] & mask).bit_count()
            is_regular = True
            for v in combo[1:]:
                if (adj_masks[v] & mask).bit_count() != d0:
                    is_regular = False
                    break
            if is_regular:
                return combo
                
    return None

def get_edges_from_masks(adj_masks, n):
    e = []
    for u in range(n):
        mask = adj_masks[u]
        for v in range(u + 1, n):
            if mask & (1 << v):
                e.append([u, v])
    return e

def construct():
    global START_TIME
    if START_TIME is None:
        START_TIME = time.time()

    deadline = START_TIME + DEADLINE - SAFETY_MARGIN
    rng = random.Random(42)
    
    # Target n=21
    n = 21
    adj_masks = [0] * n
    
    # Generate all edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            edges.append((u, v))
            
    rng.shuffle(edges)
    
    # Greedy construction
    for u, v in edges:
        if time.time() > deadline:
            break
            
        # Check if adding (u, v) creates a size-6 violation
        # Note: check_size6_violations_containing assumes edge is NOT present
        # We need to ensure it's not present. Since we build from empty and only add, it's not present.
        if not check_size6_violations_containing(adj_masks, n, u, v):
            adj_masks[u] |= (1 << v)
            adj_masks[v] |= (1 << u)
        else:
            # Skip this edge
            pass

    # Verify and repair
    # The greedy construction ensures no size-6 violations.
    # But it might have size-7+ violations.
    # Also, the greedy construction might be too restrictive and result in a small graph?
    # No, it just skips edges. The graph will be as dense as possible without size-6 regular subgraphs.
    
    # Repair loop for larger violations
    for _ in range(100):
        if time.time() > deadline:
            break
            
        viol = find_violation(adj_masks, n, min_size=7)
        if viol is None:
            break
            
        # Pick a random edge in the violating set to remove
        vset = list(viol)
        rng.shuffle(vset)
        removed = False
        for i in range(len(vset)):
            for j in range(i + 1, len(vset)):
                u, v = vset[i], vset[j]
                if adj_masks[u] & (1 << v):
                    # Remove edge
                    adj_masks[u] ^= (1 << v)
                    adj_masks[v] ^= (1 << u)
                    removed = True
                    break
            if removed:
                break
                
        if not removed:
            break

    # Final verification
    final_viol = find_violation(adj_masks, n, min_size=6)
    if final_viol is not None:
        # Fallback: try a smaller n or random search
        # For now, return best effort. If invalid, score is 0.
        # We should ensure validity.
        # Let's do a simple local search on the current graph to fix any remaining violations.
        for _ in range(50):
            if time.time() > deadline:
                break
            viol = find_violation(adj_masks, n, min_size=6)
            if viol is None:
                break
            vset = list(viol)
            rng.shuffle(vset)
            # Try removing edges
            for i in range(len(vset)):
                for j in range(i + 1, len(vset)):
                    u, v = vset[i], vset[j]
                    if adj_masks[u] & (1 << v):
                        adj_masks[u] ^= (1 << v)
                        adj_masks[v] ^= (1 << u)
                        break

    # Check if valid
    if find_violation(adj_masks, n, min_size=6) is None:
        return {'n': n, 'edges': get_edges_from_masks(adj_masks, n)}
    else:
        # Fallback to a known valid small graph if we fail
        # Random graph on n=10 is likely valid?
        # Let's just return n=10 random graph as a safe bet if 21 fails
        n_safe = 10
        adj_masks_safe = [0] * n_safe
        for u in range(n_safe):
            for v in range(u + 1, n_safe):
                if rng.random() < 0.5:
                    adj_masks_safe[u] |= (1 << v)
                    adj_masks_safe[v] |= (1 << u)
        
        # Verify safe graph
        if find_violation(adj_masks_safe, n_safe, min_size=6) is None:
            return {'n': n_safe, 'edges': get_edges_from_masks(adj_masks_safe, n_safe)}
        else:
            # Return empty graph? No, must be valid.
            # n=5 is always valid (no subset of size 6)
            return {'n': 5, 'edges': []}

if __name__ == "__main__":
    result = construct()
    print(f"Found graph with n={result['n']}, edges={len(result['edges'])}")
