"""
Dense rainbow-4-cycle edge-coloured graphs on 20 vertices.

Strategy:
1. Start with a complete graph K20 with a strong initial rainbow-4-cycle colouring
   (based on Z20 modular arithmetic to guarantee no rainbow violations).
2. The initial colouring is: colour(u, v) = (u + v) % 20 for u < v.
   This is known to avoid repeated colours in 4-cycles due to algebraic properties.
3. Since the initial graph is already valid and dense (190 edges), the only way to
   improve is... wait, K20 has 190 edges. That's the maximum possible for a simple graph.
   So if the initial colouring is valid, we're done.

Let me verify: Is colour(u,v) = (u+v) mod 20 valid for all 4-cycles in K20?

A 4-cycle is u-v-w-x-u with all distinct.
Colours: c1=(u+v)%20, c2=(v+w)%20, c3=(w+x)%20, c4=(x+u)%20.

For these to all be distinct:
c1 ≠ c2 ⟺ u+v ≠ v+w (mod 20) ⟺ u ≠ w (mod 20) ✓ (distinct vertices)
c1 ≠ c3 ⟺ u+v ≠ w+x (mod 20)
c1 ≠ c4 ⟺ u+v ≠ x+u (mod 20) ⟺ v ≠ x (mod 20) ✓
c2 ≠ c3 ⟺ v+w ≠ w+x (mod 20) ⟺ v ≠ x (mod 20) ✓
c2 ≠ c4 ⟺ v+w ≠ x+u (mod 20)
c3 ≠ c4 ⟺ w+x ≠ x+u (mod 20) ⟺ w ≠ u (mod 20) ✓

So we need: u+v ≠ w+x (mod 20) and v+w ≠ x+u (mod 20).

u+v ≠ w+x ⟺ u+v-w-x ≠ 0 (mod 20)
v+w ≠ x+u ⟺ v+w-x-u ≠ 0 (mod 20)

These are not automatically guaranteed. For example, u=0, v=1, w=2, x=19:
c1=1, c2=3, c3=21%20=1, c4=19. c1==c3. Bad!

So the simple (u+v)%20 doesn't work. Let me think of a better initial colouring.

Alternative: Use a proper edge-colouring of K20. K20 has chromatic index 19 (odd n),
so we can edge-colour with 19 colours properly. But we have 20 colours available.
A proper edge-colouring ensures adjacent edges have different colours, but 4-cycles
have non-adjacent edges that could share colours.

Better approach: Use a 1-factorization of K20. K20 has 19 perfect matchings.
Colour edge (u,v) with the matching index that contains it. Then any 4-cycle uses
edges from 4 different matchings? No, a 4-cycle u-v-w-x-u: edges (u,v) and (w,x) could
be in the same matching if u-v and w-x are in the same 1-factor. But (u,v) and (w,x)
are opposite edges in the 4-cycle. If they're in the same 1-factor, they have the same
colour, violating rainbow.

So we need a stronger structure. Let me use a different approach:

Use the fact that we can assign colours to avoid 4-cycle conflicts. Start with K20
and use local search (WalkSAT-style) to fix any 4-cycle colour conflicts while keeping
all edges. Since we start with all 190 edges, we just need to find a valid colouring.

If we can find a valid colouring for K20, we get 190 edges which is optimal.
"""
import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n = 20
    k = 20  # number of colours
    
    # Start with complete graph
    adj = np.ones((n, n), dtype=np.uint8)
    np.fill_diagonal(adj, 0)
    
    # Initial colouring: try a structured approach
    # Use colour (u * 2 - v) mod 20 or similar to break symmetry
    # Actually, let's try: for edge (u,v) with u < v, colour = (u + 2*v) % 20
    # Or better: use a Latin-square-like structure.
    
    # Let's use: colour(u, v) = (u + v) % k for the initial guess, then repair.
    colors = np.zeros((n, n), dtype=np.int32)
    for u in range(n):
        for v in range(u + 1, n):
            c = (u + v) % k
            colors[u, v] = c
            colors[v, u] = c
    
    def count_bad_cycles(adj, colors):
        """Count 4-cycles with non-rainbow colouring."""
        count = 0
        for u in range(n):
            for w in range(u + 1, n):
                common = np.where(adj[u] & adj[w])[0]
                L = len(common)
                if L < 2:
                    continue
                for i in range(L):
                    v = common[i]
                    cv_u = colors[u, v]
                    cv_w = colors[v, w]
                    for j in range(i + 1, L):
                        x = common[j]
                        cx_w = colors[w, x]
                        cx_u = colors[x, u]
                        if (cv_u == cv_w or cv_u == cx_w or cv_u == cx_u or
                            cv_w == cx_w or cv_w == cx_u or cx_w == cx_u):
                            count += 1
        return count
    
    def get_random_bad_cycle(adj, colors):
        """Return a random bad 4-cycle or None."""
        bad_list = []
        for u in range(n):
            for w in range(u + 1, n):
                common = np.where(adj[u] & adj[w])[0]
                L = len(common)
                if L < 2:
                    continue
                for i in range(L):
                    v = common[i]
                    cv_u = colors[u, v]
                    cv_w = colors[v, w]
                    for j in range(i + 1, L):
                        x = common[j]
                        cx_w = colors[w, x]
                        cx_u = colors[x, u]
                        if (cv_u == cv_w or cv_u == cx_w or cv_u == cx_u or
                            cv_w == cx_w or cv_w == cx_u or cx_w == cx_u):
                            bad_list.append((u, v, w, x))
        if not bad_list:
            return None
        return random.choice(bad_list)
    
    current_bad = count_bad_cycles(adj, colors)
    
    # WalkSAT-style repair: recolor edges to fix bad 4-cycles
    T = 5.0
    T_min = 0.05
    cooling = 0.9995
    
    while time.time() < deadline:
        if current_bad == 0:
            # Valid! K20 with 190 edges. Return it.
            break
        
        # Find a bad cycle
        bad = get_random_bad_cycle(adj, colors)
        if bad is None:
            current_bad = 0
            break
        
        u, v, w, x = bad
        cycle_edges = [(u, v), (v, w), (w, x), (x, u)]
        
        # Pick an edge from the cycle to recolor
        e_u, e_v = random.choice(cycle_edges)
        old_c = colors[e_u, e_v]
        new_c = random.randint(0, k - 1)
        if new_c == old_c:
            new_c = (new_c + 1) % k
        
        colors[e_u, e_v] = new_c
        colors[e_v, e_u] = new_c
        
        new_bad = count_bad_cycles(adj, colors)
        
        # Accept if improvement, or with probability based on temperature
        delta = new_bad - current_bad
        if delta < 0 or (T > T_min and random.random() < np.exp(-delta / T)):
            current_bad = new_bad
        else:
            # Revert
            colors[e_u, e_v] = old_c
            colors[e_v, e_u] = old_c
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # If we still have bad cycles, fall back to greedy edge removal
    if current_bad > 0:
        # Greedily remove edges to fix bad cycles
        while time.time() < deadline - 5:
            bad = get_random_bad_cycle(adj, colors)
            if bad is None:
                break
            u, v, w, x = bad
            cycle_edges = [(u, v), (v, w), (w, x), (x, u)]
            e_u, e_v = random.choice(cycle_edges)
            adj[e_u, e_v] = 0
            adj[e_v, e_u] = 0
            colors[e_u, e_v] = 0
            colors[e_v, e_u] = 0
            current_bad = count_bad_cycles(adj, colors)
    
    # Construct output
    result = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u, v] == 1:
                c = int(colors[u, v])
                if c == 0 and colors[u, v] == 0:
                    # Edge was removed but color is 0; assign a valid color
                    c = random.randint(0, k - 1)
                result.append([u, v, c])
    
    return result

# EVOLVE-BLOCK-END
