import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    
    # Precompute popcount for speed? 
    # Python 3.10+ has int.bit_count() which is very fast.
    
    def count_violations(G, k, M, sizes):
        """
        Counts how many subsets S of existing vertices of given sizes
        would form a regular induced subgraph when v (with mask M) is added.
        """
        violations = 0
        vertices = list(range(k))
        
        for size in sizes:
            if size > k:
                continue
            for S in combinations(vertices, size):
                S_mask = 0
                for u in S:
                    S_mask |= (1 << u)
                
                # d is the degree of v in the subgraph induced by S U {v}
                d = (S_mask & M).bit_count()
                
                # Check if all u in S have degree d in G[S U {v}]
                # deg(u) in G[S U {v}] = deg(u) in G[S] + (1 if u is neighbor of v else 0)
                is_regular = True
                for u in S:
                    # deg(u) in G[S]
                    deg_u = (G[u] & S_mask).bit_count()
                    # connection to v
                    conn_v = 1 if (M & (1 << u)) else 0
                    if deg_u + conn_v != d:
                        is_regular = False
                        break
                
                if is_regular:
                    violations += 1
                    if violations > 0:
                        # We just need to know if it's 0 or not, 
                        # or minimize. 
                        # For speed, we can return early if we just want validity?
                        # But we want to minimize. 
                        # However, if we find 1, we know it's > 0.
                        # To minimize, we need the exact count? 
                        # Or just find a 0?
                        # Strategy: Find a mask with 0 violations.
                        # If none, pick one with min violations.
                        # For speed in the "find 0" phase, we can break early?
                        # No, if we break early, we don't know if it's the min.
                        # But if we find ANY violation, this mask is "bad".
                        # We can just return a large number if we just want to filter out bad masks quickly?
                        # Let's return the count, but break if count > some threshold?
                        # Actually, if we are just looking for a valid mask, we can return 1 immediately?
                        # But we want to compare candidates.
                        # Let's just count.
                        pass
        
        return violations

    def verify_graph(G, n):
        # Full verification for the final graph
        # Check all subsets of size >= 6
        # This is expensive, so only do it at the end or occasionally.
        vertices = list(range(n))
        for size in range(6, n + 1):
            for S in combinations(vertices, size):
                S_mask = 0
                for u in S:
                    S_mask |= (1 << u)
                
                # Check if regular
                # Get degree of first vertex
                first_deg = -1
                is_regular = True
                for u in S:
                    deg_u = (G[u] & S_mask).bit_count()
                    if first_deg == -1:
                        first_deg = deg_u
                    elif deg_u != first_deg:
                        is_regular = False
                        break
                
                if is_regular:
                    return False
        return True

    best_n = 5
    best_graph = {'n': 5, 'edges': []}
    
    # Try multiple restarts
    for attempt in range(10):
        if time.time() - start_time > 110:
            break
            
        random.seed(attempt)
        G = [0] * 5
        edges = []
        # Initialize 5 vertices with random edges
        for i in range(5):
            for j in range(i+1, 5):
                if random.random() < 0.5:
                    G[i] |= (1 << j)
                    G[j] |= (1 << i)
                    edges.append([i, j])
        
        current_n = 5
        success = True
        
        # Sizes of S to check: 5, 6, 7
        # This covers new subgraphs of size 6, 7, 8
        check_sizes = [5, 6, 7]
        
        for target_n in range(6, 23):
            if time.time() - start_time > 110:
                success = False
                break
                
            k = current_n
            best_mask = None
            min_violations = float('inf')
            
            # Try a bunch of random masks
            # Increase tries as k increases?
            num_tries = 100
            
            for _ in range(num_tries):
                cand_mask = random.getrandbits(k)
                v = count_violations(G, k, cand_mask, check_sizes)
                if v < min_violations:
                    min_violations = v
                    best_mask = cand_mask
                    if v == 0:
                        break # Found a perfect mask
            
            if min_violations > 0:
                # No perfect mask found. 
                # We can still try the best one, but it might lead to an invalid final graph.
                # If we are close to the end, maybe we should stop?
                # Or accept it and hope the final verification passes?
                # The final verification will catch it.
                # But if we accept a "bad" mask, we might get stuck later.
                # Let's accept the best mask anyway to try to maximize n, 
                # but keep track that we have violations.
                # Actually, if min_violations > 0, the graph *will* have a regular subgraph 
                # of size 6, 7, or 8. So the final graph will be INVALID.
                # Therefore, we MUST find a mask with 0 violations to be valid.
                # If we can't, we cannot extend further with a valid graph.
                # So we must stop.
                success = False
                break
            
            # Add the vertex with best_mask (which has 0 violations for sizes 5,6,7)
            G.append(best_mask)
            for i in range(k):
                if (best_mask >> i) & 1:
                    G[i] |= (1 << k)
                    edges.append([i, k])
            
            current_n += 1
            
            # Update best if this is better
            # We don't verify every step because it's expensive.
            # We verify at the end of the attempt.
            
        if success and current_n > best_n:
            # Final verification
            if verify_graph(G, current_n):
                best_n = current_n
                best_graph = {'n': current_n, 'edges': edges.copy()}
                if current_n >= 20:
                    return best_graph
            else:
                # The graph is invalid. 
                # This can happen if we missed a regular subgraph of size > 8 
                # that was not detected by the incremental check.
                # Or if the incremental check was too optimistic.
                # We should try to repair or just discard this attempt.
                # Discarding is safer.
                pass

    return best_graph
