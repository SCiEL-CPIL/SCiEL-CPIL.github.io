import random
import itertools
import time
import sys

def _find_violation(n, adj_mask):
    """
    Find a regular induced subgraph of size >= 6.
    Returns (True, subset_tuple) if found, (False, None) otherwise.
    Uses bitmask representation for adjacency.
    """
    vertices = list(range(n))
    
    # We check sizes 6 through n.
    # For efficiency, we can stop early if we find one.
    # Since C(n,k) is largest around k=n/2, and n is small (<=22),
    # checking all sizes is feasible if we are careful.
    
    # Optimization: 
    # A regular induced subgraph of size k has degree d.
    # The complement is also regular with degree k-1-d.
    # This symmetry doesn't help much directly but good to know.
    
    # We iterate k from 6 to n.
    for k in range(6, n + 1):
        for subset in itertools.combinations(vertices, k):
            mask_S = 0
            for v in subset:
                mask_S |= (1 << v)
            
            degs = []
            first_deg = -1
            is_regular = True
            
            for v in subset:
                # Count neighbors of v within subset
                # adj_mask[v] contains bits for all neighbors of v
                # We need intersection with mask_S
                # Note: adj_mask[v] does NOT include v itself (simple graph, no self-loops)
                # So popcount(adj_mask[v] & mask_S) is the degree of v in the induced subgraph.
                d = bin(adj_mask[v] & mask_S).count('1')
                
                if first_deg == -1:
                    first_deg = d
                elif d != first_deg:
                    is_regular = False
                    break
            
            if is_regular:
                return True, subset
                
    return False, None

def _construct_from_seed(seed):
    """
    Try to find a valid graph using local search starting from a random seed.
    This is a different strategy from pure random restarts: it uses a repair heuristic.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s for safety
    
    # We aim for n=21 or 22.
    # Let's try n=21 first, as it's the next step up from the known 20.
    # Actually, the task says "Maximize n".
    # Known: G(6) > 20 is possible (n=20 graph exists).
    # Goal: find n=21.
    
    target_n = 21
    
    # Strategy: Start with a random graph, then iteratively fix violations.
    # A violation is a regular induced subgraph S of size k >= 6.
    # To break it, we can add or remove an edge within S (or incident to S? No, induced subgraph depends only on internal edges).
    # So we must change an edge (u,v) where u,v in S.
    # If we add an edge (u,v) in S, degrees of u and v in S increase by 1.
    # If we remove an edge (u,v) in S, degrees of u and v in S decrease by 1.
    
    # Heuristic: 
    # 1. Generate random graph.
    # 2. Find a violation S.
    # 3. If no violation, return success.
    # 4. If violation, pick an edge (u,v) in S to flip.
    #    - If S is regular of degree d, flipping an edge changes degrees of u,v.
    #    - We want to make it non-regular.
    #    - Simply flipping a random edge in S might work, or we can be smarter.
    #    - Alternatively, flip an edge that minimizes the number of *new* violations? That's expensive.
    #    - Let's try: pick a random edge in S to flip.
    
    best_graph = None
    best_violation_count = float('inf')
    
    # Try multiple random starts
    for attempt in range(50):
        if time.time() > deadline:
            break
            
        # Initialize random graph
        adj_mask = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        # Local search loop
        for step in range(1000):
            if time.time() > deadline:
                break
                
            found, subset = _find_violation(target_n, adj_mask)
            
            if not found:
                # Success!
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj_mask[i] & (1 << j):
                            edges.append([i, j])
                return {'n': target_n, 'edges': edges}
            
            # Found a violation. Try to fix it.
            # Pick a random edge within the violating subset to flip.
            if len(subset) < 2:
                continue
                
            # Get list of edges in subset
            edges_in_subset = []
            for i_idx in range(len(subset)):
                for j_idx in range(i_idx + 1, len(subset)):
                    u = subset[i_idx]
                    v = subset[j_idx]
                    edges_in_subset.append((u, v))
            
            if not edges_in_subset:
                continue
                
            u, v = random.choice(edges_in_subset)
            
            # Flip edge (u,v)
            if adj_mask[u] & (1 << v):
                # Remove
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                # Add
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)
                
    # If we didn't find a valid graph for n=21, try n=20 (which is known to exist)
    # But we want to maximize n. If n=21 fails, we should return the best we found.
    # The current program returns n=5 if it fails completely.
    # We should try to find a valid graph for n=20 as a fallback.
    
    # Fallback: find a valid graph for n=20
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        target_n = 20
        adj_mask = [0] * target_n
        for i in range(target_n):
            for j in range(i + 1, target_n):
                if random.random() < 0.5:
                    adj_mask[i] |= (1 << j)
                    adj_mask[j] |= (1 << i)
        
        for step in range(1000):
            if time.time() > deadline:
                break
                
            found, subset = _find_violation(target_n, adj_mask)
            
            if not found:
                edges = []
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if adj_mask[i] & (1 << j):
                            edges.append([i, j])
                return {'n': target_n, 'edges': edges}
            
            if len(subset) < 2:
                continue
                
            edges_in_subset = []
            for i_idx in range(len(subset)):
                for j_idx in range(i_idx + 1, len(subset)):
                    u = subset[i_idx]
                    v = subset[j_idx]
                    edges_in_subset.append((u, v))
            
            if not edges_in_subset:
                continue
                
            u, v = random.choice(edges_in_subset)
            
            if adj_mask[u] & (1 << v):
                adj_mask[u] &= ~(1 << v)
                adj_mask[v] &= ~(1 << u)
            else:
                adj_mask[u] |= (1 << v)
                adj_mask[v] |= (1 << u)

    # Final fallback: return a small valid graph (e.g., n=5)
    # Any graph on 5 vertices is valid since we only check subsets of size >= 6.
    # So an empty graph on 5 vertices is valid.
    return {'n': 5, 'edges': []}

def construct():
    """
    Entry point.
    """
    # Set a seed for reproducibility? No, random is fine.
    # But maybe we should try a few different strategies or seeds.
    # The local search approach is probabilistic.
    
    # Try to find n=21 or n=20.
    # The function _construct_from_seed does this.
    
    result = _construct_from_seed(0)
    
    return result

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found graph with n={res['n']}, edges={len(res['edges'])}")
    # Verify
    n = res['n']
    edges = res['edges']
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    found, subset = _find_violation(n, adj_mask)
    if found:
        print(f"ERROR: Found violation: {subset}")
    else:
        print("Graph is valid.")
