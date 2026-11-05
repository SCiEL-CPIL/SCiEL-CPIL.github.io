import time
from math import gcd

N = 512
DEADLINE = 82.0  # Safety margin

def construct():
    start_time = time.time()
    
    # 1. Build conflict graph
    adj = [0] * N
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            g = gcd(a, b)
            if (a * b) // g > N:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    
    # 2. Find strong initial solution via multiple strategies
    # Strategy A: Divisors of highly composite numbers
    def get_divisors(M):
        divs = []
        for d in range(1, M + 1):
            if M % d == 0:
                divs.append(d)
        return divs
    
    best_init_size = 0
    best_init_set = [1]
    
    # Try all M from 512 down
    for M in range(512, 0, -1):
        divs = get_divisors(M)
        if len(divs) > best_init_size:
            best_init_size = len(divs)
            best_init_set = divs[:]
    
    # Strategy B: Greedy by number of compatible partners
    def count_compatible(x, current_set):
        cnt = 0
        for y in current_set:
            g = gcd(x, y)
            if (x * y) // g <= N:
                cnt += 1
        return cnt
    
    greedy_set = [1]
    remaining = list(range(2, 513))
    while remaining:
        best_a = None
        best_score = -1
        for a in remaining:
            # Count how many in greedy_set are compatible
            score = count_compatible(a, greedy_set)
            # Also count how many in remaining are compatible with a (future potential)
            if score == len(greedy_set):  # Fully compatible
                # Prefer smaller numbers (more likely to be compatible with others)
                if best_a is None or a < best_a:
                    best_a = a
                    best_score = score
        if best_a is None:
            break
        greedy_set.append(best_a)
        remaining.remove(best_a)
    
    if len(greedy_set) > best_init_size:
        best_init_size = len(greedy_set)
        best_init_set = greedy_set[:]
    
    # Strategy C: Greedy with different orderings
    for start_val in [1, 2, 3, 4, 6, 12, 24]:
        g_set = [start_val]
        rem = [x for x in range(1, 513) if x != start_val]
        changed = True
        while changed:
            changed = False
            best_a = None
            best_compat = -1
            for a in rem:
                ok = True
                for b in g_set:
                    gg = gcd(a, b)
                    if (a * b) // gg > N:
                        ok = False
                        break
                if ok:
                    # Count compatibility with remaining
                    compat = 0
                    for c in rem:
                        if c == a:
                            continue
                        gg2 = gcd(a, c)
                        if (a * c) // gg2 <= N:
                            compat += 1
                    if compat > best_compat:
                        best_compat = compat
                        best_a = a
            if best_a is not None:
                g_set.append(best_a)
                rem.remove(best_a)
                changed = True
        if len(g_set) > best_init_size:
            best_init_size = len(g_set)
            best_init_set = g_set[:]
    
    # Reorder vertices by degree descending for better B&B
    degrees = [adj[i].bit_count() for i in range(N)]
    order = sorted(range(N), key=lambda x: -degrees[x])
    
    new_index = [0] * N
    for new_idx, orig_idx in enumerate(order):
        new_index[orig_idx] = new_idx
    
    adj_new = [0] * N
    for i in range(N):
        orig_i = order[i]
        mask = 0
        m = adj[orig_i]
        while m:
            lsb = m & -m
            orig_j = lsb.bit_length() - 1
            new_j = new_index[orig_j]
            mask |= (1 << new_j)
            m ^= lsb
        adj_new[i] = mask

    # 3. Optimized greedy coloring bound
    # Precompute a static ordering for coloring (by degree)
    # Use iterative greedy coloring with early termination
    def get_color_bound(R):
        colors = 0
        remaining = R
        while remaining:
            color_class = 0
            pool = remaining
            while pool:
                v_bit = pool & -pool
                v = v_bit.bit_length() - 1
                color_class |= v_bit
                pool &= ~v_bit
                pool &= ~adj_new[v]
            remaining &= ~color_class
            colors += 1
            # Early termination: if we already exceed what's needed
            if colors >= 50:  # Very loose upper bound
                return colors
        return colors

    # 4. Set up initial solution in new indexing
    init_mask = 0
    for val in best_init_set:
        orig_idx = val - 1
        new_idx = new_index[orig_idx]
        init_mask |= (1 << new_idx)
    
    # Verify validity
    valid = True
    m = init_mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        if adj_new[v] & init_mask:
            valid = False
            break
        m ^= lsb
    
    best_size = len(best_init_set) if valid else 1
    best_set = best_init_set[:] if valid else [1]
    
    # 5. Branch and Bound with tighter bounds
    def solve(R, size, current_set_indices):
        nonlocal best_size, best_set, start_time
        
        if time.time() - start_time > DEADLINE:
            return
        
        if R == 0:
            if size > best_size:
                best_size = size
                best_set = [order[i] + 1 for i in current_set_indices]
            return
        
        # Quick bound: max possible additions
        remaining_count = R.bit_count()
        if size + remaining_count <= best_size:
            return
        
        bound = get_color_bound(R)
        if size + bound <= best_size:
            return
        
        # Select vertex: highest degree in R
        best_v = -1
        max_deg = -1
        temp = R
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            deg = (adj_new[v] & R).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= lsb
        
        if best_v == -1:
            return
        
        # Branch 1: Include best_v
        new_R_incl = R & ~(adj_new[best_v] | (1 << best_v))
        current_set_indices.append(best_v)
        solve(new_R_incl, size + 1, current_set_indices)
        current_set_indices.pop()
        
        # Branch 2: Exclude best_v
        new_R_excl = R & ~(1 << best_v)
        solve(new_R_excl, size, current_set_indices)
        
    # Start B&B from the full set
    initial_mask = (1 << N) - 1
    solve(initial_mask, 0, [])
    
    # 6. Local improvement: try to add more elements to best_set
    if best_set:
        best_set_set = set(best_set)
        improved = True
        while improved:
            improved = False
            for a in range(1, 513):
                if a in best_set_set:
                    continue
                ok = True
                for b in best_set_set:
                    g = gcd(a, b)
                    if (a * b) // g > 512:
                        ok = False
                        break
                if ok:
                    best_set_set.add(a)
                    improved = True
        if len(best_set_set) > best_size:
            best_size = len(best_set_set)
            best_set = list(best_set_set)
    
    if not best_set:
        return [1]
    return best_set

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
