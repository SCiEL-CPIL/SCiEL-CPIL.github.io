import time
import random
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    # Start with Paley graph on 41 vertices
    q = 41
    n = q
    adj = [0] * n
    
    # Quadratic residues mod 41
    residues = set()
    for x in range(1, q):
        residues.add((x * x) % q)
        
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)

    def get_violation():
        # Check for K5 or IK5
        # Using bitsets for speed
        for u in range(n):
            for v in range(u+1, n):
                if not (adj[u] & (1 << v)): continue
                for w in range(v+1, n):
                    if not (adj[v] & (1 << w)): continue
                    # u-v-w connected
                    # Need x, y such that u-v-w-x-y is a K5
                    # x must connect to u, v, w
                    # y must connect to u, v, w, x
                    # Candidates for x: adj[u] & adj[v] & adj[w]
                    # We need x > w
                    # And then y > x in candidates & adj[x]
                    
                    # Let's just do the 5-tuple check for simplicity and reliability, 
                    # but add a time check inside? No, get_violation should be fast.
                    pass
        
        # Fallback to 5-tuple if the above is too complex to implement correctly in time
        # But 5-tuple is O(n^5).
        # Let's use the 5-tuple check but optimize by breaking early.
        for combo in combinations(range(n), 5):
            u, v, w, x, y = combo
            # Check K5
            if (adj[u] & (1 << v)) and (adj[u] & (1 << w)) and (adj[u] & (1 << x)) and (adj[u] & (1 << y)) and \
               (adj[v] & (1 << w)) and (adj[v] & (1 << x)) and (adj[v] & (1 << y)) and \
               (adj[w] & (1 << x)) and (adj[w] & (1 << y)) and \
               (adj[x] & (1 << y)):
                return ('K5', combo)
            
            # Check IK5
            if not (adj[u] & (1 << v)) and not (adj[u] & (1 << w)) and not (adj[u] & (1 << x)) and not (adj[u] & (1 << y)) and \
               not (adj[v] & (1 << w)) and not (adj[v] & (1 << x)) and not (adj[v] & (1 << y)) and \
               not (adj[w] & (1 << x)) and not (adj[w] & (1 << y)) and \
               not (adj[x] & (1 << y)):
                return ('IK5', combo)
        return None

    def flip_edge(u, v):
        if adj[u] & (1 << v):
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
        else:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)

    # Repair Phase
    # We want to reach a valid state.
    # Use Simulated Annealing
    T = 1.0
    cooling = 0.99
    stuck = 0
    
    while time.time() < deadline - 20:
        viol = get_violation()
        if viol is None:
            break
        
        vtype, verts = viol
        # Pick random edge in violation
        u, v = random.choice(list(combinations(verts, 2)))
        
        # Calculate delta? Hard to calculate delta without counting all violations.
        # Just flip.
        flip_edge(u, v)
        
        T *= cooling
        if T < 0.1:
            T = 1.0

    # If we are valid, try to add vertices
    if get_violation() is None:
        while n < 46 and time.time() < deadline - 5:
            # Get K4s and IK4s
            k4s = []
            ik4s = []
            for quad in combinations(range(n), 4):
                is_k4 = True
                is_ik4 = True
                for u, v in combinations(quad, 2):
                    if adj[u] & (1 << v):
                        is_ik4 = False
                    else:
                        is_k4 = False
                if is_k4:
                    k4s.append(quad)
                if is_ik4:
                    ik4s.append(quad)
            
            # Local search for new vertex
            x = [random.randint(0, 1) for _ in range(n)]
            success = False
            for _ in range(5000):
                viol_k4 = None
                viol_ik4 = None
                for c in k4s:
                    if all(x[u] == 1 for u in c):
                        viol_k4 = c
                        break
                if viol_k4 is None:
                    for c in ik4s:
                        if all(x[u] == 0 for u in c):
                            viol_ik4 = c
                            break
                
                if viol_k4 is None and viol_ik4 is None:
                    success = True
                    break
                
                if viol_k4 is not None:
                    u = random.choice(viol_k4)
                    x[u] = 0
                else:
                    u = random.choice(viol_ik4)
                    x[u] = 1
                    
            if success:
                new_v = n
                adj.append(0)
                for u in range(n):
                    if x[u]:
                        adj[new_v] |= (1 << u)
                        adj[u] |= (1 << new_v)
                n += 1
            else:
                break

    # Extract edges
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i] & (1 << j):
                edges.append((i, j))
                
    return n, edges

# EVOLVE-BLOCK-END
