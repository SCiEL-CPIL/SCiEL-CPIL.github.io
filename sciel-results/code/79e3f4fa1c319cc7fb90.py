import time
import random
import sys

def construct():
    def instance(n, p, seed):
        rng = random.Random(seed)
        adj = [0] * n
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return adj

    def count_bits(x):
        return bin(x).count("1")

    def greedy_partition(adj, n, seed):
        """
        Fast greedy construction of a valid clique partition.
        Returns list of cliques (lists of vertices).
        """
        rng = random.Random(seed)
        rem_adj = adj[:]
        cliques = []
        
        # Precompute initial degrees for faster start? 
        # We will just scan.
        
        while True:
            # Find vertex with max degree in remaining graph
            max_deg = 0
            start_v = -1
            for u in range(n):
                deg = count_bits(rem_adj[u])
                if deg > max_deg:
                    max_deg = deg
                    start_v = u
            
            if max_deg == 0:
                break
            
            # Build a clique starting from start_v
            # We use a randomized greedy strategy: at each step, pick a neighbor
            # that is connected to all current clique members and has high degree.
            clique = [start_v]
            candidates = rem_adj[start_v]
            
            while candidates:
                # Pick best candidate
                # To add randomness, we can shuffle candidates or pick from top-k
                # For speed, we just pick max degree, but randomize tie-breaking
                best_cand = -1
                best_deg = -1
                c = candidates
                while c:
                    lsb = c & -c
                    v = lsb.bit_length() - 1
                    deg = count_bits(rem_adj[v])
                    # Add small random noise to break ties and diversify
                    score = deg + rng.random() * 0.1
                    if score > best_deg:
                        best_deg = score
                        best_cand = v
                    c ^= lsb
                
                if best_cand == -1:
                    break
                
                clique.append(best_cand)
                # Update candidates: intersection with neighbors of best_cand
                candidates &= rem_adj[best_cand]
            
            # Remove edges in clique
            for u in clique:
                for v in clique:
                    if u < v:
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
            
            cliques.append(clique)
        
        return cliques

    def local_search_improve(adj, n, cliques, seed, time_limit):
        """
        Improve the partition by trying to merge cliques or re-pack edges.
        Strategy: 
        1. Check if any two cliques can be merged into a larger clique.
        2. If not, try to "break" a clique and redistribute its edges to potentially form fewer cliques.
        
        For simplicity and speed, we focus on:
        - Merging: If C1 and C2 are cliques, and every vertex in C1 is connected to every vertex in C2, 
          then C1 U C2 is a clique. We can replace C1, C2 with C1 U C2.
        - This is a safe local move that reduces clique count by 1.
        
        We also try a "re-greedy" on a subset of edges if we break a clique, but that's complex.
        Let's stick to merging first, then maybe a simple swap.
        """
        rng = random.Random(seed)
        current_cliques = [c[:] for c in cliques]
        
        # Helper to check if union of two cliques is a clique
        def is_union_clique(c1, c2, adj):
            for u in c1:
                for v in c2:
                    if not (adj[u] & (1 << v)):
                        return False
            return True
        
        improved = True
        start_time = time.time()
        
        while improved:
            if time.time() - start_time > time_limit:
                break
            
            improved = False
            # Shuffle order to avoid local minima
            indices = list(range(len(current_cliques)))
            rng.shuffle(indices)
            
            for i_idx in range(len(indices)):
                i = indices[i_idx]
                c1 = current_cliques[i]
                if not c1:
                    continue
                for j_idx in range(i_idx + 1, len(indices)):
                    j = indices[j_idx]
                    c2 = current_cliques[j]
                    if not c2:
                        continue
                    
                    # Try to merge c1 and c2
                    if is_union_clique(c1, c2, adj):
                        new_clique = c1 + c2
                        # Remove i and j, add new_clique
                        # Be careful with indices if we remove
                        # We will rebuild the list
                        new_list = []
                        for k in range(len(current_cliques)):
                            if k != i and k != j:
                                new_list.append(current_cliques[k])
                        new_list.append(new_clique)
                        current_cliques = new_list
                        improved = True
                        # Restart loop since structure changed
                        break
                
                if improved:
                    break
        
        return current_cliques

    def solve_instance(adj, n, seed, time_budget):
        best_cliques = None
        best_score = float('inf')
        
        # Run multiple restarts of greedy + local search
        start_time = time.time()
        restart = 0
        
        while time.time() - start_time < time_budget:
            # 1. Greedy construction
            cliques = greedy_partition(adj, n, seed + restart)
            
            # 2. Local search improvement (merging)
            # Give local search a small time slice
            ls_time = min(0.5, (time_budget - (time.time() - start_time)) * 0.5)
            if ls_time > 0.1:
                improved_cliques = local_search_improve(adj, n, cliques, seed + restart, ls_time)
            else:
                improved_cliques = cliques
            
            score = len(improved_cliques)
            if score < best_score:
                best_score = score
                best_cliques = improved_cliques
            
            # Early stop if we found a very good solution? 
            # Hard to know optimal. Just keep searching.
            restart += 1
            
        if best_cliques is None:
            # Fallback
            best_cliques = greedy_partition(adj, n, seed)
            
        return best_cliques

    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    adj_1 = instance(32, 0.6, 601)
    adj_2 = instance(64, 0.55, 602)
    
    # Allocate time roughly proportional to complexity (n^2)
    # n1=32, n2=64. n2 is 4x harder.
    # Let's give 25% time to n1 and 75% to n2.
    time_for_1 = 20
    time_for_2 = 80
    
    # Solve instance 1
    cliques_1 = solve_instance(adj_1, 32, 100, time_for_1)
    
    # Check time
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    if remaining < 10:
        # Fallback for instance 2 if running out of time
        cliques_2 = greedy_partition(adj_2, 64, 200)
    else:
        # Solve instance 2
        cliques_2 = solve_instance(adj_2, 64, 200, remaining - 5)
    
    return [cliques_1, cliques_2]
