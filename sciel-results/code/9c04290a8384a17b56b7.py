import time
import numpy as np
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115.0

    def remaining():
        return deadline - time.time()

    # ---------------- girth verification ----------------
    def check_girth(n, edges, g, time_limit):
        if n == 0:
            return -1
        adj = [[] for _ in range(n)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)

        # Check degree
        for i in range(n):
            if len(adj[i]) != k:
                return -1

        min_cycle = float('inf')
        for s in range(n):
            if time.time() > time_limit:
                return -1
            dist = [-1] * n
            parent = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                node = queue[qi]
                qi += 1
                for nb in adj[node]:
                    if nb == parent[node]:
                        continue
                    if dist[nb] != -1:
                        cycle_len = dist[node] + dist[nb] + 1
                        if cycle_len < min_cycle:
                            min_cycle = cycle_len
                        if cycle_len < g:
                            return min_cycle
                    else:
                        dist[nb] = dist[node] + 1
                        parent[nb] = node
                        queue.append(nb)
            if min_cycle < g:
                return min_cycle
        return min_cycle

    # ---------------- Cayley Graph Construction ----------------
    # For k=3: Cayley graph of a group G with 3 involutive generators {a, b, c}
    # where a^2 = b^2 = c^2 = e. Girth >= g means no non-trivial reduced word
    # of length < g in {a,b,c} evaluates to identity.
    # 
    # Strategy: Build a group presentation incrementally using a Todd-Coxeter
    # style approach, or use a randomized search for permutation representations.
    #
    # Key insight: A 3-regular graph with girth >= 16 can be viewed as the
    # Schreier graph of a permutation representation. We search for permutations
    # (a, b, c) in S_n such that:
    #   - Each is an involution (a^2 = b^2 = c^2 = identity)
    #   - The group they generate is transitive (connected graph)
    #   - No word of length < 16 in {a,b,c} (with no consecutive same letters)
    #     evaluates to identity on any point (girth condition)
    
    def cayley_graph_search():
        if k != 3 or g != 16:
            return None

        # For n vertices, we need 3 involutions a, b, c in S_n
        # such that the Schreier graph has girth >= 16.
        #
        # The Moore bound for k=3, g=16 is 766.
        # Let's try n values from 768 upward (must be even for 3-regular).
        #
        # Actually, for a 3-regular graph, n must be even.
        # Moore bound: 2 + 3*(1 + 2 + 4 + ... + 2^7) = 2 + 3*255 = 767
        # So n >= 768 (next even).
        
        # Precompute all reduced words of length up to 7 (half of girth 16)
        # A reduced word has no consecutive identical generators.
        # For girth >= 16, we need: for every reduced word w of length 1..7,
        # w != identity, AND for every reduced word w of length 8,
        # w != identity on any point (actually we need length < 16, so up to 15).
        #
        # Wait: girth >= 16 means no cycle of length < 16. A cycle of length L
        # corresponds to a word of length L that evaluates to identity, where
        # the word is reduced (no backtracking). So we need no reduced word of
        # length 1,2,...,15 to evaluate to identity.
        #
        # But checking all words up to length 15 is expensive. Key observation:
        # If a reduced word of length L < 16 evaluates to identity, then there's
        # a shorter one or it's a fundamental relation. We can use the fact that
        # for girth >= 16, the ball of radius 7 around any vertex is a tree.
        #
        # Efficient check: BFS from each vertex, verify no collision within depth 7.
        # For a 3-regular graph, ball of radius r has 1 + 3*(2^r - 1) vertices
        # if it's a tree. For r=7: 1 + 3*127 = 382.
        # So n must be >= 2*382 = 764 (since two balls of radius 7 around adjacent
        # vertices overlap). Actually the Moore bound is exactly this.
        
        # Strategy: Randomly generate 3 involutions and check girth.
        # To make this efficient, we use a "growing" approach:
        # Start with a partial assignment and extend, checking constraints.
        
        def random_involution(n, rng):
            """Generate a random involution (fixed-point-free or with fixed points)."""
            perm = list(range(n))
            for i in range(n):
                if perm[i] == i:
                    # Find a partner
                    j = rng.randint(i+1, n-1)
                    if perm[j] == j:
                        perm[i], perm[j] = perm[j], perm[i]
                    else:
                        # Find another unpaired
                        for jj in range(i+1, n):
                            if perm[jj] == jj:
                                perm[i], perm[jj] = perm[jj], perm[i]
                                break
            # Verify it's an involution
            for i in range(n):
                if perm[perm[i]] != i:
                    return None
            return perm

        def check_schreier_girth(n, gens, g, time_limit):
            """Check that the Schreier graph has girth >= g."""
            adj = [[] for _ in range(n)]
            for gen in gens:
                for i in range(n):
                    j = gen[i]
                    if i != j:
                        adj[i].append(j)
            
            # Check degree
            for i in range(n):
                if len(adj[i]) != k:
                    return -1
            
            # BFS from each vertex to find shortest cycle
            min_cycle = float('inf')
            for s in range(n):
                if time.time() > time_limit:
                    return -1
                dist = [-1] * n
                parent = [-1] * n
                dist[s] = 0
                queue = [s]
                qi = 0
                while qi < len(queue):
                    node = queue[qi]
                    qi += 1
                    for nb in adj[node]:
                        if nb == parent[node]:
                            continue
                        if dist[nb] != -1:
                            cycle_len = dist[node] + dist[nb] + 1
                            if cycle_len < min_cycle:
                                min_cycle = cycle_len
                                if cycle_len < g:
                                    return min_cycle
                        else:
                            dist[nb] = dist[node] + 1
                            parent[nb] = node
                            queue.append(nb)
            return min_cycle

        def build_from_generators(n, gens):
            """Build edge list from 3 involutions."""
            edges = set()
            for gen in gens:
                for i in range(n):
                    j = gen[i]
                    if i < j:
                        edges.add((i, j))
            return list(edges)

        # Try different n values starting from Moore bound
        # n must be even for 3-regular graph
        for n_try in [768, 770, 772, 774, 776, 778, 780, 782, 784, 786, 788, 790, 792, 794, 796, 798, 800]:
            if remaining() < 15:
                break
            
            rng = random.Random(n_try * 12345 + 678)
            
            # Try multiple random attempts
            max_attempts = 50
            for attempt in range(max_attempts):
                if remaining() < 10:
                    break
                
                # Generate 3 random involutions
                gens = []
                valid = True
                for _ in range(3):
                    gen = None
                    for _ in range(20):  # retries
                        gen = random_involution(n_try, rng)
                        if gen is not None:
                            break
                    if gen is None:
                        valid = False
                        break
                    gens.append(gen)
                
                if not valid:
                    continue
                
                # Quick check: verify they're involutions
                for gen in gens:
                    for i in range(n_try):
                        if gen[gen[i]] != i:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check girth (this is the expensive part)
                gc = check_schreier_girth(n_try, gens, g, deadline - 5)
                if gc >= g:
                    edges = build_from_generators(n_try, gens)
                    # Final verification
                    gc2 = check_girth(n_try, edges, g, deadline - 2)
                    if gc2 >= g:
                        return (n_try, edges)
            
            # If simple random search fails, try a more structured approach
            # for this n: use the "alternating" pattern
            if remaining() < 10:
                break

        return None

    # Try Cayley graph approach first
    res = cayley_graph_search()
    if res is not None:
        return res

    # Fallback: voltage lift (original approach, simplified)
    def voltage_lift_search():
        if k != 3 or g != 16:
            return None
        
        pet_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
            (5, 7), (7, 9), (9, 6), (6, 8), (8, 5),
            (0, 5), (1, 6), (2, 7), (3, 8), (4, 9)
        ]
        
        n_base = 10
        edge_index = {}
        for i, (u, v) in enumerate(pet_edges):
            edge_index[(u, v)] = i
            edge_index[(v, u)] = i
            
        adj = [[] for _ in range(n_base)]
        for u, v in pet_edges:
            adj[u].append(v)
            adj[v].append(u)

        # Generate constraints from non-backtracking walks
        max_L = g - 1
        walk_vectors = set()
        
        for start_v in range(n_base):
            stack = [(start_v, -1, 0, [0]*15)]
            while stack:
                current, prev, length, counts = stack.pop()
                
                if length >= max_L:
                    continue
                
                for next_v in adj[current]:
                    if next_v == prev:
                        continue
                    e_idx = edge_index[(current, next_v)]
                    u, v = pet_edges[e_idx]
                    d = 1 if current == u else -1
                    
                    new_counts = counts.copy()
                    new_counts[e_idx] += d
                    new_l = length + 1
                    
                    if next_v == start_v and new_l >= 5:
                        vec = tuple(new_counts)
                        if any(x != 0 for x in vec):
                            first_nz = -1
                            for i in range(15):
                                if vec[i] != 0:
                                    first_nz = i
                                    break
                            if first_nz != -1 and vec[first_nz] < 0:
                                vec = tuple(-x for x in vec)
                            walk_vectors.add(vec)
                    
                    stack.append((next_v, current, new_l, new_counts))
        
        if not walk_vectors:
            return None
            
        C = np.array(list(walk_vectors), dtype=np.int64)
        
        def is_prime(n):
            if n < 2: return False
            if n < 4: return True
            if n % 2 == 0 or n % 3 == 0: return False
            i = 5
            while i * i <= n:
                if n % i == 0 or n % (i + 2) == 0: return False
                i += 6
            return True

        def solve_voltage(m, time_limit):
            v = np.random.randint(1, m, size=15).astype(np.int64)
            
            def cost(v):
                residues = np.mod(C @ v, m)
                return np.sum(residues == 0)
            
            best_v = v.copy()
            best_cost = cost(v)
            
            if best_cost == 0:
                return best_v
            
            T = 1.0
            cooling = 0.999
            start_t = time.time()
            
            while time.time() - start_t < time_limit:
                if remaining() < 5:
                    break
                
                j = random.randint(0, 14)
                old_val = v[j]
                v[j] = random.randint(1, m-1)
                
                new_cost = cost(v)
                delta = new_cost - best_cost
                
                if delta < 0 or random.random() < np.exp(-delta / T):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_v = v.copy()
                        if best_cost == 0:
                            return best_v
                else:
                    v[j] = old_val
                
                T *= cooling
            
            return best_v if best_cost == 0 else None
        
        m = 79
        while m < 300:
            if remaining() < 10:
                break
            
            if is_prime(m):
                t_limit = min(5.0, remaining() / 5.0)
                
                v = solve_voltage(m, t_limit)
                
                if v is not None:
                    n = 10 * m
                    edges = []
                    for u, v_base in pet_edges:
                        e_idx = edge_index[(u, v_base)]
                        vol = v[e_idx]
                        for layer in range(m):
                            u_lift = u * m + layer
                            v_lift = v_base * m + (layer + vol) % m
                            if u_lift < v_lift:
                                edges.append((u_lift, v_lift))
                            else:
                                edges.append((v_lift, u_lift))
                    
                    gc = check_girth(n, edges, g, deadline - 2)
                    if gc >= g:
                        return (n, edges)
            
            m += 1
            
        return None

    res2 = voltage_lift_search()
    if res2 is not None:
        return res2

    return (0, [])

if __name__ == "__main__":
    n, edges = construct(3, 16)
    print(f"n={n}")
