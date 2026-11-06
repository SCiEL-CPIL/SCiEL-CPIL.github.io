"""Strong edge colourings via bitmask SA + greedy."""
import time
import random
from collections import defaultdict

def construct():
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill

    def instance(n, d, seed):
        rng = random.Random(seed)
        degree, edges = [0] * n, set()
        for _ in range(4 * n * d):
            u, v = rng.randrange(n), rng.randrange(n)
            e = (min(u, v), max(u, v))
            if u != v and degree[u] < d and degree[v] < d and e not in edges:
                edges.add(e)
                degree[u] += 1
                degree[v] += 1
        return sorted(edges)

    def build_conflict_bitmasks(edges, n):
        """Build adjacency and conflict bitmasks."""
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        m = len(edges)
        incident = [set() for _ in range(n)]
        for idx, (u, v) in enumerate(edges):
            incident[u].add(idx)
            incident[v].add(idx)
        
        # Build conflict bitmasks
        conflict = [0] * m
        for idx, (u, v) in enumerate(edges):
            mask = 0
            # Edges sharing u or v
            for j in incident[u]:
                if j != idx:
                    mask |= (1 << j)
            for j in incident[v]:
                if j != idx:
                    mask |= (1 << j)
            # Edges with endpoint connected to u or v
            for w in adj[u]:
                for j in incident[w]:
                    if j != idx:
                        mask |= (1 << j)
            for w in adj[v]:
                for j in incident[w]:
                    if j != idx:
                        mask |= (1 << j)
            conflict[idx] = mask
        
        # Also build adjacency list from bitmasks for iteration
        conflict_list = []
        for i in range(m):
            lst = []
            mask = conflict[i]
            while mask:
                low = mask & (-mask)
                j = low.bit_length() - 1
                lst.append(j)
                mask ^= low
            conflict_list.append(lst)
        
        return conflict, conflict_list, adj

    def greedy_first_fit(conflict, m, order=None):
        """Greedy colouring using bitmask conflict checks."""
        if order is None:
            order = list(range(m))
        colour = [-1] * m
        # colour_mask[c] = bitmask of edges with colour c
        colour_masks = []
        for i in order:
            c = 0
            while c < len(colour_masks) and (conflict[i] & colour_masks[c]):
                c += 1
            if c == len(colour_masks):
                colour_masks.append(0)
            colour[i] = c
            colour_masks[c] |= (1 << i)
        return colour, len(colour_masks)

    def sa_coloring(conflict, conflict_list, m, init_colour, init_ncols, max_time=30.0, seed=42):
        """Simulated annealing to reduce colour count."""
        rng = random.Random(seed)
        t_start = time.time()
        t_end = t_start + max_time
        
        colour = init_colour[:]
        ncols = init_ncols
        
        # Build colour masks
        def build_colour_masks(col, nc):
            masks = [0] * nc
            for i in range(m):
                masks[col[i]] |= (1 << i)
            return masks
        
        colour_masks = build_colour_masks(colour, ncols)
        best_colour = colour[:]
        best_ncols = ncols
        best_time = t_start
        
        # SA parameters
        T = float(ncols)
        T_min = 0.01
        cooling = 0.9995
        
        # Precompute edge degrees for ordering
        deg = [len(conflict_list[i]) for i in range(m)]
        
        step = 0
        while time.time() < t_end:
            step += 1
            
            # Check for improvement
            if ncols < best_ncols:
                best_ncols = ncols
                best_colour = colour[:]
                best_time = time.time()
            
            # Pick a random edge
            i = rng.randrange(m)
            old_c = colour[i]
            
            # Try to assign a new colour
            if ncols > 1:
                # Try to remove a colour: pick from highest colour class
                # Find edges in the highest colour class
                high_edges = []
                for j in range(m):
                    if colour[j] == ncols - 1:
                        high_edges.append(j)
                
                if high_edges:
                    i = rng.choice(high_edges)
                    old_c = colour[i]
                    
                    # Try to reassign to a lower colour
                    available = []
                    for c in range(ncols - 1):
                        if not (conflict[i] & colour_masks[c]):
                            available.append(c)
                    
                    if available:
                        new_c = rng.choice(available)
                    else:
                        # Can't reassign, try random edge
                        i = rng.randrange(m)
                        old_c = colour[i]
                        available = []
                        for c in range(ncols):
                            if c != old_c and not (conflict[i] & colour_masks[c]):
                                available.append(c)
                        if not available:
                            continue
                        new_c = rng.choice(available)
                else:
                    continue
            else:
                continue
            
            # Check if new_c is available
            if conflict[i] & colour_masks[new_c]:
                continue
            
            # Accept or reject based on SA
            delta = 0  # same number of colours if we stay within range
            
            # Accept with probability based on temperature
            if delta <= 0 or rng.random() < pow(2.718281828, -delta / T):
                # Apply move
                colour_masks[old_c] &= ~(1 << i)
                colour_masks[new_c] |= (1 << i)
                colour[i] = new_c
                
                # Check if we can reduce ncols
                if old_c == ncols - 1:
                    # Check if colour class ncols-1 is now empty
                    empty = True
                    for j in range(m):
                        if colour[j] == ncols - 1:
                            empty = False
                            break
                    if empty:
                        ncols -= 1
                        colour_masks.pop()
                        # Renumber colours if needed (optional, skip for simplicity)
            
            # Cool down
            T = max(T * cooling, T_min)
            
            # Occasionally restart temperature if stuck
            if step % 5000 == 0 and ncols >= best_ncols:
                T = float(ncols) * 0.5
        
        # Return best found
        # Verify best_colour is valid
        valid = True
        for i in range(m):
            mask = conflict[i]
            ci = best_colour[i]
            # Check no conflicting edge has same colour
            # We need to check all j in conflict[i]
            for j in conflict_list[i]:
                if best_colour[j] == ci:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fall back to init
            return init_colour, init_ncols
        
        return best_colour, best_ncols

    def verify_and_fix(colour, conflict, conflict_list, m):
        """Verify colouring is valid; if not, fix by greedy."""
        # Check validity
        for i in range(m):
            ci = colour[i]
            for j in conflict_list[i]:
                if j > i and colour[j] == ci:
                    return None  # Invalid
        return colour

    results = []
    specs = ((40, 4, 101), (60, 5, 102), (80, 6, 103))
    
    for spec in specs:
        if time.time() > deadline - 10:
            # Not enough time, use greedy for remaining
            edges = instance(*spec)
            n = spec[0]
            m = len(edges)
            conflict, conflict_list, adj = build_conflict_bitmasks(edges, n)
            colour, ncols = greedy_first_fit(conflict, m)
            results.append(colour)
            continue
        
        edges = instance(*spec)
        n = spec[0]
        m = len(edges)
        conflict, conflict_list, adj = build_conflict_bitmasks(edges, n)
        
        # Get multiple greedy initializations with different orderings
        best_colour = None
        best_ncols = float('inf')
        
        # Natural order
        c1, nc1 = greedy_first_fit(conflict, m)
        if nc1 < best_ncols:
            best_ncols = nc1
            best_colour = c1[:]
        
        # Reverse order
        c2, nc2 = greedy_first_fit(conflict, m, order=list(range(m - 1, -1, -1)))
        if nc2 < best_ncols:
            best_ncols = nc2
            best_colour = c2[:]
        
        # Degree-ordered
        deg = [len(conflict_list[i]) for i in range(m)]
        order = sorted(range(m), key=lambda i: -deg[i])
        c3, nc3 = greedy_first_fit(conflict, m, order=order)
        if nc3 < best_ncols:
            best_ncols = nc3
            best_colour = c3[:]
        
        # Random orders
        rng = random.Random(12345)
        for _ in range(10):
            ord = list(range(m))
            rng.shuffle(ord)
            c, nc = greedy_first_fit(conflict, m, order=ord)
            if nc < best_ncols:
                best_ncols = nc
                best_colour = c[:]
        
        # Verify best
        fixed = verify_and_fix(best_colour, conflict, conflict_list, m)
        if fixed is None:
            # Fallback to natural greedy
            best_colour, best_ncols = greedy_first_fit(conflict, m)
        
        # Run SA with multiple seeds
        time_remaining = deadline - time.time()
        sa_time = max(1.0, min(30.0, time_remaining / 3))
        
        for seed in range(8):
            if time.time() > deadline - 5:
                break
            c, nc = sa_coloring(conflict, conflict_list, m, best_colour, best_ncols,
                               max_time=sa_time, seed=seed)
            if nc < best_ncols:
                best_ncols = nc
                best_colour = c[:]
        
        # Final verification
        fixed = verify_and_fix(best_colour, conflict, conflict_list, m)
        if fixed is None:
            # Fallback
            best_colour, best_ncols = greedy_first_fit(conflict, m)
        
        results.append(best_colour)
    
    return results
