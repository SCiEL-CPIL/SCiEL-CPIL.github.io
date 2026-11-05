import time
import random
from itertools import combinations

def construct():
    n = 18
    deadline = time.time() + 115.0
    random.seed(12345)

    # Start with Turán graph T(18,3): parts of size 6
    # Part assignment: vertex i is in part i//6
    # Edge between u,v exists iff part(u) != part(v)
    
    def make_turan3():
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                pi, pj = i // 6, j // 6
                if pi != pj:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    def is_k4_free(adj):
        # Check no 4 vertices form K4
        # K4 = 6 edges among 4 vertices
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if all(((adj[a] >> v) & 1) for v in (b, c, d)) and \
               all(((adj[b] >> v) & 1) for v in (a, c, d)) and \
               all(((adj[c] >> v) & 1) for v in (a, b, d)):
                return False
        return True

    def largest_triangle_free(adj):
        """Exact branch-and-bound for largest triangle-free induced subgraph."""
        full = (1 << n) - 1
        best = [0]
        
        # Greedy initial bound
        degs = [bin(adj[i]).count('1') for i in range(n)]
        order = sorted(range(n), key=lambda x: -degs[x])
        greedy = 0
        for v in order:
            if not (adj[v] & greedy):
                greedy |= (1 << v)
        best[0] = bin(greedy).count('1')
        
        # Try multiple random orders
        for _ in range(5):
            order = list(range(n))
            random.shuffle(order)
            order.sort(key=lambda x: -degs[x])
            g = 0
            for v in order:
                if not (adj[v] & g):
                    g |= (1 << v)
            sz = bin(g).count('1')
            if sz > best[0]:
                best[0] = sz
        
        def bnb(S, C, remaining):
            if bin(S).count('1') + remaining <= best[0]:
                return
            if remaining == 0:
                if bin(S).count('1') > best[0]:
                    best[0] = bin(S).count('1')
                return
            
            # Pick vertex from C with max degree within C
            best_v = -1
            best_d = -1
            cm = C
            while cm:
                lsb = cm & -cm
                v = lsb.bit_length() - 1
                d = bin(adj[v] & C).count('1')
                if d > best_d:
                    best_d = d
                    best_v = v
                cm ^= lsb
            
            v = best_v
            vb = 1 << v
            
            # Branch 1: exclude v
            bnb(S, C & ~vb, remaining - 1)
            
            # Branch 2: include v (must be triangle-free with S)
            if not (adj[v] & S):
                # New candidate set: vertices in C that are not adjacent to v
                newC = C & ~vb & ~adj[v]
                new_remaining = bin(newC).count('1')
                bnb(S | vb, newC, new_remaining)
        
        bnb(0, full, n)
        return best[0]

    def try_add_edge(adj, u, v):
        """Check if adding edge (u,v) preserves K4-free property."""
        if (adj[u] >> v) & 1:
            return True  # already exists
        # Adding edge (u,v) creates K4 if there exist common neighbors x,y
        # such that {u,v,x,y} forms K4. This means x,y are common neighbors of u,v
        # and x-y is an edge, and x is adjacent to v, y is adjacent to u.
        # Actually: K4 on {u,v,x,y} requires all 6 edges.
        # We have edges among {x,y} from common neighbors, and need:
        # u-x, u-y (x,y are common neighbors so yes), v-x, v-y (yes by definition),
        # and x-y must exist.
        common = adj[u] & adj[v]
        # Check if common contains an edge
        cm = common
        while cm:
            lsb = cm & -cm
            x = lsb.bit_length() - 1
            if (common & (1 << x)) and (adj[x] & (common ^ (1 << x))):
                # There's an edge in common
                # Actually simpler: check if any pair in common is an edge
                pass
            cm ^= lsb
        # Simpler: check if the induced subgraph on common contains any edge
        cm2 = common
        while cm2:
            lsb = cm2 & -cm2
            x = lsb.bit_length() - 1
            if adj[x] & (common & ~(1 << x)):
                return False
            cm2 ^= lsb
        return True

    def try_remove_edge(adj, u, v):
        if not ((adj[u] >> v) & 1):
            return True
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        return True

    def restore_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Local search: try to reduce largest TF by modifying edges
    def local_search(adj, current_score, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Random edge modifications
            for _ in range(200):
                if time.time() > time_limit:
                    break
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v:
                    continue
                has_edge = (adj[u] >> v) & 1
                
                if has_edge:
                    # Try removing
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if random.random() < 0.3:
                        new_score = largest_triangle_free(adj)
                        if new_score < current_score:
                            current_score = new_score
                            improved = True
                        else:
                            restore_edge(adj, u, v)
                else:
                    # Try adding
                    if try_add_edge(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        if random.random() < 0.3:
                            new_score = largest_triangle_free(adj)
                            if new_score < current_score:
                                current_score = new_score
                                improved = True
                            else:
                                adj[u] &= ~(1 << v)
                                adj[v] &= ~(1 << u)
        return current_score

    # Main: start from Turán, then local search
    adj = make_turan3()
    score = largest_triangle_free(adj)
    print(f"Turán initial: {score}")
    
    # Local search to improve
    time_limit = deadline - 5.0
    score = local_search(adj, score, time_limit)
    print(f"After local search: {score}")
    
    # If still above some threshold, try more aggressive search
    if score > 11 and time.time() < deadline - 10.0:
        # Try random perturbations
        for _ in range(50):
            if time.time() > deadline - 5.0:
                break
            adj2 = make_turan3()
            # Randomly remove some edges
            for _ in range(random.randint(1, 5)):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and (adj2[u] >> v) & 1:
                    adj2[u] &= ~(1 << v)
                    adj2[v] &= ~(1 << u)
            s2 = largest_triangle_free(adj2)
            if s2 < score:
                score = s2
                adj = adj2[:]
                print(f"Perturbation improved: {score}")
    
    # Build edge list
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                edges.append([i, j])
    
    print(f"Final: score={score}, edges={len(edges)}")
    return edges
