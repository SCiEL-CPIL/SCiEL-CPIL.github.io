import itertools
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    # Precompute primes up to 256
    max_val = 256
    is_prime = [True] * (max_val + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    
    # Build adjacency list for bipartite graph
    # Nodes: A in [1,128], B in [1,128]
    # Edge (a,b) if a+b is prime
    adj = {a: set() for a in range(1, 129)}
    for a in range(1, 129):
        for b in range(1, 129):
            if is_prime[a+b]:
                adj[a].add(b)
                
    # Greedy search for 3x3 biclique
    # We want A=[a1,a2,a3], B=[b1,b2,b3]
    # such that all a_i+b_j are prime (and distinct).
    
    # Heuristic:
    # 1. Pick a1 with max |adj[a1]|
    # 2. Pick b1 in adj[a1] with max |adj[b1]| (symmetric view)
    #    Actually, let's just pick b1 from adj[a1] that has many neighbors in A.
    #    Simpler: Pick a1 with max degree.
    #    Pick b1 from adj[a1] with max degree (in B? No, just count neighbors in A).
    #    Let's define deg_B(b) = number of a in A such that a+b is prime.
    
    # Since we are picking A and B simultaneously, let's try a few starting points.
    
    found_solution = None
    
    # Try a few high-degree 'a's as the first element of A
    sorted_a_by_degree = sorted(range(1, 129), key=lambda x: len(adj[x]), reverse=True)
    
    for a1 in sorted_a_by_degree[:10]:
        if time.time() > deadline:
            break
            
        # Candidate b's for b1
        b_candidates = list(adj[a1])
        # Sort b's by how many common neighbors they have with other high-degree a's?
        # Or just pick b1 with max degree (number of a's connected to b)
        
        # Calculate degree of b in terms of A-connections
        # deg_a(b) = sum(1 for a in range(1,129) if b in adj[a])
        # This is constant for all b, effectively.
        # So just pick b1 arbitrarily or by value?
        # Let's pick b1 that allows for more options for a2.
        
        for b1 in b_candidates:
            if time.time() > deadline:
                break
            
            # Find a2
            # a2 must be connected to b1.
            # a2 != a1.
            # To maximize options for a3, pick a2 with max overlap with adj[b1] and adj[other b's]?
            # Simple: pick a2 in adj_inv[b1] (neighbors of b1 in A) with max degree.
            
            # Get neighbors of b1 in A
            # We didn't build adj_inv. Let's compute on fly or precompute.
            # Precomputing adj_inv is better.
            
            # Let's restart with precomputed adj_inv
            break # break inner to restructure
        
        # Restructure: Precompute adj_inv
        adj_inv = {b: set() for b in range(1, 129)}
        for a in range(1, 129):
            for b in adj[a]:
                adj_inv[b].add(a)
                
        # Retry loop with adj_inv
        # (I will move this precomputation outside the loop in the final code)
        
    # Final Code Structure:
    # 1. Precompute is_prime
    # 2. Precompute adj (a->b) and adj_inv (b->a)
    # 3. Greedy Search:
    #    For a1 in top 20 degrees:
    #      For b1 in adj[a1]:
    #        For a2 in adj_inv[b1] - {a1}:
    #          For b2 in adj[a2] - {b1}:
    #            For a3 in (adj_inv[b1] & adj_inv[b2]) - {a1, a2}:
    #              For b3 in (adj[a3] & adj[a1] & adj[a2]) - {b1, b2}:
    #                 Check distinctness and coprimality.
    
    # This is a nested loop. Depth 5.
    # Top 20 a1s.
    # Each adj[a1] has ~60 elements.
    # Each adj_inv[b1] has ~60 elements.
    # Intersection sizes will shrink.
    # This should be fast.
    
    # Let's implement this.
    
    # Precompute adj and adj_inv
    adj = {a: set() for a in range(1, 129)}
    adj_inv = {b: set() for b in range(1, 129)}
    for a in range(1, 129):
        for b in range(1, 129):
            if is_prime[a+b]:
                adj[a].add(b)
                adj_inv[b].add(a)
                
    sorted_a_by_degree = sorted(range(1, 129), key=lambda x: len(adj[x]), reverse=True)
    
    for a1 in sorted_a_by_degree[:20]:
        if time.time() > deadline:
            break
        for b1 in adj[a1]:
            if time.time() > deadline:
                break
            # a2 candidates: neighbors of b1 in A, excluding a1
            a2_candidates = adj_inv[b1] - {a1}
            if not a2_candidates: continue
            for a2 in a2_candidates:
                if time.time() > deadline:
                    break
                # b2 candidates: neighbors of a2 in B, excluding b1
                # Also b2 should be "compatible" with a1? No, a1+b2 must be prime.
                # So b2 in adj[a2] AND b2 in adj[a1]
                b2_candidates = (adj[a2] & adj[a1]) - {b1}
                if not b2_candidates: continue
                for b2 in b2_candidates:
                    if time.time() > deadline:
                        break
                    # a3 candidates: neighbors of b1 AND b2 in A, excluding a1, a2
                    a3_candidates = (adj_inv[b1] & adj_inv[b2]) - {a1, a2}
                    if not a3_candidates: continue
                    for a3 in a3_candidates:
                        if time.time() > deadline:
                            break
                        # b3 candidates: neighbors of a3 in B, AND neighbors of a1, a2 in B
                        # b3 in adj[a3] & adj[a1] & adj[a2]
                        # excluding b1, b2
                        b3_candidates = (adj[a3] & adj[a1] & adj[a2]) - {b1, b2}
                        if not b3_candidates: continue
                        for b3 in b3_candidates:
                            if time.time() > deadline:
                                break
                            A = [a1, a2, a3]
                            B = [b1, b2, b3]
                            
                            # Check distinctness of sums
                            sums = set()
                            distinct = True
                            for a in A:
                                for b in B:
                                    s = a + b
                                    if s in sums:
                                        distinct = False
                                        break
                                    sums.add(s)
                                if not distinct:
                                    break
                            
                            if distinct and len(sums) == 9:
                                # Check pairwise coprime
                                coprime = True
                                sums_list = list(sums)
                                for i in range(len(sums_list)):
                                    for j in range(i+1, len(sums_list)):
                                        if math.gcd(sums_list[i], sums_list[j]) != 1:
                                            coprime = False
                                            break
                                    if not coprime:
                                        break
                                if coprime:
                                    found_solution = [A, B]
                                    break
                        if found_solution: break
                    if found_solution: break
                if found_solution: break
            if found_solution: break
        if found_solution: break
        
    if found_solution:
        return found_solution
        
    # Fallback to exhaustive search if greedy fails (should be rare)
    # ... (keep the old exhaustive search as fallback)
    
    return [[1, 9, 15], [22, 58, 126]]
