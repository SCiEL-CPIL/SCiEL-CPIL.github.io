"""Combined approach: 3-partite Turán structure with vertex-swap local search and bitmask K4-free checking."""
import time
import random
import math
import sys

def construct():
    """Construct a K4-free graph on 18 vertices minimizing the max triangle-free induced subgraph."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    N = 18
    full_mask = (1 << N) - 1
    
    # Helper: Check if a bitmask represents an independent set in graph defined by adj (list of bitmasks)
    def is_independent(mask, adj):
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            if adj[v] & (mask ^ lsb):
                return False
            m ^= lsb
        return True
    
    # Helper: Check if adding edge (u,v) keeps graph K4-free
    def can_add_edge(adj, u, v):
        if (adj[u] >> v) & 1:
            return True
        common = adj[u] & adj[v]
        return is_independent(common, adj)
    
    # Exact B&B for largest triangle-free induced subgraph
    def largest_triangle_free(adj, time_limit=None):
        best = [0]
        
        # Greedy heuristic
        for _ in range(5):
            order = list(range(N))
            random.shuffle(order)
            S = 0
            for v in order:
                if is_independent(adj[v] & S, adj):
                    S |= (1 << v)
            sz = bin(S).count('1')
            if sz > best[0]:
                best[0] = sz
        
        if best[0] == N:
            return N
        
        # Branch and bound
        def bnb(S_mask, C_mask):
            if time_limit and time.time() > time_limit:
                return
            sz = bin(S_mask).count('1')
            
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            if not C_mask:
                if sz > best[0]:
                    best[0] = sz
                return
            
            # Prune: remove candidates that cannot be added
            new_C = 0
            c = C_mask
            while c:
                lsb = c & -c
                v = lsb.bit_length() - 1
                if is_independent(adj[v] & S_mask, adj):
                    new_C |= lsb
                c ^= lsb
            
            if not new_C:
                if sz > best[0]:
                    best[0] = sz
                return
            
            C_mask = new_C
            if sz + bin(C_mask).count('1') <= best[0]:
                return
            
            lsb = C_mask & -C_mask
            v = lsb.bit_length() - 1
            
            bnb(S_mask | lsb, C_mask ^ lsb)
            bnb(S_mask, C_mask ^ lsb)
        
        bnb(0, full_mask)
        return best[0]
    
    # Get edge list from adjacency bitmasks
    def get_edge_list(adj):
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if (adj[i] >> j) & 1:
                    edges.append([i, j])
        return edges
    
    # 3-partite Turán graph: partition vertices into 3 parts, edges between different parts
    # Any triangle-free induced subgraph can use vertices from at most 2 parts
    # (3 vertices from 3 different parts form a triangle in a complete 3-partite graph)
    
    def make_3partite_adj(partition):
        """Create adjacency from a 3-partition of vertices."""
        adj = [0] * N
        part_of = [0] * N
        for p in range(3):
            for v in partition[p]:
                part_of[v] = p
        for i in range(N):
            for j in range(i + 1, N):
                if part_of[i] != part_of[j]:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj
    
    # For a 3-partite graph, compute largest triangle-free induced subgraph
    # A triangle-free set S in a 3-partite graph: if S has vertices from all 3 parts,
    # then any triple (one from each part) forms a triangle. So S can only use 2 parts.
    # But wait - we need to be more careful. In a complete 3-partite graph,
    # a triangle-free induced subgraph can contain vertices from at most 2 parts.
    # So the answer is max(|A∪B|, |A∪C|, |B∪C|) where A,B,C are the parts.
    def largest_triangle_free_3partite(partition):
        """For a complete 3-partite graph, largest triangle-free induced subgraph."""
        a, b, c = len(partition[0]), len(partition[1]), len(partition[2])
        return max(a + b, a + c, b + c)
    
    # But our graph may not be perfectly 3-partite after modifications.
    # Use the general method for verification.
    
    # Verify K4-free
    def is_k4_free(adj):
        for i in range(N):
            for j in range(i + 1, N):
                if not ((adj[i] >> j) & 1):
                    continue
                for k in range(j + 1, N):
                    if not ((adj[i] >> k) & 1) or not ((adj[j] >> k) & 1):
                        continue
                    for l in range(k + 1, N):
                        if ((adj[i] >> l) & 1) and ((adj[j] >> l) & 1) and ((adj[k] >> l) & 1):
                            return False
        return True
    
    random.seed(42)
    
    # Start with balanced 3-partite Turán: parts of size 6,6,6
    # Largest triangle-free = max(6+6, 6+6, 6+6) = 12
    best_partition = [[i for i in range(6)], [i for i in range(6, 12)], [i for i in range(12, 18)]]
    best_adj = make_3partite_adj(best_partition)
    best_score = largest_triangle_free(best_adj, deadline - 5.0)
    
    # Local search: try moving a vertex from one part to another
    def evaluate_partition(partition):
        """Evaluate a 3-partition by computing largest triangle-free subgraph."""
        adj = make_3partite_adj(partition)
        return largest_triangle_free(adj, deadline - 5.0)
    
    # Vertex-swap local search
    def local_search_3partite(partition, best_score, deadline):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for p in range(3):
                if time.time() > deadline:
                    break
                for v in partition[p][:]:  # Copy to allow modification
                    if time.time() > deadline:
                        break
                    for q in range(3):
                        if q == p:
                            continue
                        # Move v from part p to part q
                        new_partition = [list(part[i]) for part in partition]
                        new_partition[p].remove(v)
                        new_partition[q].append(v)
                        
                        # Quick estimate: for complete 3-partite, score = max of sums of 2 parts
                        a, b, c = len(new_partition[0]), len(new_partition[1]), len(new_partition[2])
                        quick_score = max(a + b, a + c, b + c)
                        
                        if quick_score >= best_score:
                            continue
                        
                        # Full evaluation
                        score = evaluate_partition(new_partition)
                        if score < best_score:
                            best_score = score
                            partition = new_partition
                            improved = True
        
        return partition, best_score
    
    # Run local search from balanced start
    best_partition, best_score = local_search_3partite(best_partition, best_score, deadline - 10.0)
    best_adj = make_3partite_adj(best_partition)
    
    # Random restarts with different initial partitions
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        
        # Generate random 3-partition
        random_partition = [[], [], []]
        for v in range(N):
            random_partition[random.randint(0, 2)].append(v)
        
        # Ensure no part is empty (degenerate cases)
        for p in range(3):
            if not random_partition[p]:
                # Move a vertex from the largest part
                max_p = max(range(3), key=lambda x: len(random_partition[x]))
                random_partition[p].append(random_partition[max_p].pop())
        
        score = evaluate_partition(random_partition)
        if score < best_score:
            best_score = score
            best_partition = random_partition
            best_partition, best_score = local_search_3partite(best_partition, best_score, deadline - 10.0)
            best_adj = make_3partite_adj(best_partition)
        
        # Also try perturbing current best
        if attempts % 3 == 0 and time.time() < deadline - 15.0:
            perturbed = [list(part) for part in best_partition]
            num_moves = random.randint(1, 3)
            for _ in range(num_moves):
                p = random.randint(0, 2)
                if not perturbed[p]:
                    continue
                v = random.choice(perturbed[p])
                perturbed[p].remove(v)
                q = random.randint(0, 2)
                perturbed[q].append(v)
            
            score = evaluate_partition(perturbed)
            if score < best_score:
                best_score = score
                best_partition = perturbed
                best_partition, best_score = local_search_3partite(best_partition, best_score, deadline - 10.0)
                best_adj = make_3partite_adj(best_partition)
    
    # Final: also try some non-3-partite K4-free graphs via edge additions
    # Start from best 3-partite and try adding edges within parts (which were non-edges)
    if time.time() < deadline - 20.0:
        adj = make_3partite_adj(best_partition)
        part_of = [0] * N
        for p in range(3):
            for v in best_partition[p]:
                part_of[v] = p
        
        # Try adding edges within the same part (these don't create K4 if done carefully)
        # Actually, adding an edge within a part of a 3-partite graph:
        # Could create K4 if there are two other vertices in other parts connected to both
        # Let's be safe and just try edge flips on the 3-partite base
        
        for i in range(N):
            if time.time() > deadline - 10.0:
                break
            for j in range(i + 1, N):
                if time.time() > deadline - 10.0:
                    break
                old_val = (adj[i] >> j) & 1
                
                if old_val == 0:
                    # Try adding edge
                    if not can_add_edge(adj, i, j):
                        continue
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                else:
                    # Try removing edge
                    adj[i] &= ~(1 << j)
                    adj[j] &= ~(1 << i)
                
                score = largest_triangle_free(adj, deadline - 10.0)
                
                if score < best_score:
                    best_score = score
                    best_adj = adj[:]
                else:
                    # Revert
                    if old_val == 0:
                        adj[i] &= ~(1 << j)
                        adj[j] &= ~(1 << i)
                    else:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
    
    return get_edge_list(best_adj)
