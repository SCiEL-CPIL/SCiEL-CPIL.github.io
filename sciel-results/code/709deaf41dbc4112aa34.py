import time
import numpy as np
import itertools

def construct():
    t0 = time.time()
    deadline = t0 + 110.0

    def is_valid_coloring(n, colors):
        if len(colors) != n * (n - 1) // 2:
            return False
        adj = np.zeros((n, n), dtype=np.int8)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 3:
                    return False
                adj[i, j] = c
                adj[j, i] = c
                idx += 1
        
        for c in range(4):
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i, j] != c:
                        continue
                    for k in range(j + 1, n):
                        if adj[i, k] == c and adj[j, k] == c:
                            return False
        return True

    def try_group_construction_48():
        # Group G = Z_2^4 x Z_3
        # Size 48
        # Indices: i = 16*a + x, x in 0..15, a in 0..2
        # Addition: (x1, a1) + (x2, a2) = (x1^x2, (a1+a2)%3)
        
        # Precompute addition table
        add_table = np.zeros((48, 48), dtype=np.int32)
        for i in range(48):
            a1 = i // 16
            x1 = i % 16
            for j in range(48):
                a2 = j // 16
                x2 = j % 16
                ax = x1 ^ x2
                aa = (a1 + a2) % 3
                add_table[i, j] = 16 * aa + ax

        # Parameters: c[d][delta_sym]
        # d in 0..4 (popcount)
        # delta_sym in 0..1 (0 for delta=0, 1 for delta=1,2)
        # Total 10 parameters
        
        # Precompute for each pair (i, j) the (d, delta_sym)
        # i < j
        # diff = j - i in group? No, we need the "difference" vector.
        # In abelian group, diff = j + i (since -i = i in Z2, but -a in Z3 is 3-a).
        # Wait, the coloring is usually defined on differences.
        # col(i, j) = f(j - i).
        # In Z2, j-i = j+i.
        # In Z3, j-i = j+i mod 3? No, j-i mod 3.
        # So we need to calculate the group difference.
        
        # Let's precompute the "type" of each edge (i, j)
        # type = (d, delta_sym)
        edge_types = {}
        for i in range(48):
            a1 = i // 16
            x1 = i % 16
            for j in range(i + 1, 48):
                a2 = j // 16
                x2 = j % 16
                # Difference
                dx = x1 ^ x2
                da = (a2 - a1) % 3
                d = bin(dx).count('1')
                if da == 0:
                    delta_sym = 0
                else:
                    delta_sym = 1
                edge_types[(i, j)] = (d, delta_sym)

        # Iterate over all 4^10 mappings
        # Use itertools.product
        # To speed up, we can break early if a color set is not triangle-free.
        
        # Precompute S_c sets for a given mapping?
        # No, mapping changes.
        
        # We can iterate over mappings and check validity.
        # Optimization: Check each color independently?
        # A triangle of color c requires a,b,c in S_c.
        # So if any S_c contains a triangle, the coloring is invalid.
        
        # We can precompute the list of edges for each (d, delta_sym) type?
        # No, S_c is a set of group elements (differences).
        # The set of differences for color c is determined by the mapping.
        # Let mapping be a dict: (d, delta_sym) -> color.
        # Then S_c = { g in G | f(g) = c }.
        
        # We need to check if S_c is "sum-free" in the sense of no a+b+c=0?
        # No, no a,b,a+b in S_c.
        
        # Let's precompute the "triangle triples" for the group G.
        # A triple (a,b) forms a triangle if a!=0, b!=0, a+b!=0 and a,b,a+b are in the same color set.
        # We can precompute all pairs (a,b) that form a potential triangle.
        # There are 48*47 pairs.
        # For each mapping, we just check if any such pair has all 3 elements in the same S_c.
        
        potential_triangles = []
        for a in range(1, 48):
            for b in range(1, 48):
                s = add_table[a, b]
                if s != 0:
                    potential_triangles.append((a, b, s))
        
        # Now iterate mappings
        # 10 parameters, 4 values each.
        # We can use a recursive search or simple loop.
        # 4^10 = 1,048,576.
        # For each, 48*47/2 ~ 1000 checks? No, len(potential_triangles) ~ 48*47 - 48 ~ 2200.
        # 1M * 2200 = 2.2B ops. Too slow in Python.
        
        # Optimization:
        # For a fixed mapping, build S[4] as bitmasks.
        # For each color c, check if there is a triangle in S[c].
        # We can precompute for each color c, the set of "bad" pairs?
        # No, the mapping defines S[c].
        
        # Better Optimization:
        # Iterate over mappings.
        # For each mapping, compute S[4] bitmasks.
        # For each c, check if S[c] has a triangle.
        # Check: for a in S[c], for b in S[c], if a+b in S[c].
        # Using bitmasks:
        #   For a fixed a, B = S[c] shifted by -a.
        #   Intersection = S[c] & B.
        #   If Intersection has any b such that b!=0 and a+b!=0, fail.
        
        # How to shift a bitmask?
        # We can precompute `shift_table[mask_val][shift]`? No, mask is 48 bits.
        # We can precompute for each a, the permutation of indices.
        # perm_a[i] = i + a (mod G).
        # Then shifted_mask = 0
        # for i in range(48): if S[c] has bit i, set bit perm_a[i] in shifted_mask.
        # This is O(48) per a.
        # 48 a's * 48 ops = 2304 ops per color.
        # 4 colors * 2304 = 9216 ops per mapping.
        # 1M * 9216 = 9.2B ops. Still too slow.
        
        # We need to prune.
        # Many mappings will fail on the first color.
        # We can check colors one by one.
        
        # Let's try a random sampling of mappings first to see if any work.
        # If random sampling finds one, we are good.
        # If not, we might need a smarter search.
        
        # Actually, the 48-vertex graph is known to exist.
        # Let's try a few specific known mappings or random ones.
        
        # Random sampling
        import random
        for _ in range(10000):
            if time.time() > deadline - 10:
                break
            # Generate random mapping
            mapping = {}
            for d in range(5):
                for ds in range(2):
                    mapping[(d, ds)] = random.randint(0, 3)
            
            # Build S[4]
            S = [0] * 4
            for i in range(48):
                a = i // 16
                x = i % 16
                # This i is a group element.
                # We need to determine the "type" of this element relative to 0?
                # No, the coloring is defined by the difference.
                # So S[c] contains the group element g if f(g) = c.
                # f(g) is determined by popcount(x_g) and delta_g.
                # For a group element g=(x,a), the "difference" from 0 is g.
                # So we classify g by (popcount(x), a_sym).
                # a_sym = 0 if a==0, else 1.
                
                d = bin(x).count('1')
                ds = 0 if a == 0 else 1
                c = mapping[(d, ds)]
                S[c] |= (1 << i)
            
            # Check validity
            valid = True
            for c in range(4):
                mask = S[c]
                if mask == 0:
                    continue
                # Check for triangle
                # Iterate a in mask
                # We can just iterate all a in 0..47
                for a in range(48):
                    if not (mask & (1 << a)):
                        continue
                    if a == 0:
                        continue
                    # Compute B = mask shifted by -a
                    # b is in B if (b+a) is in mask
                    # So b = (b+a) - a
                    # We want to check if there exists b in mask such that b+a in mask.
                    # Let's just iterate b in mask?
                    # Or use the shift logic.
                    # For simplicity, let's just check all b in mask
                    # This is O(|S_c|^2)
                    # |S_c| ~ 12. 144 checks.
                    # 48 * 144 = 6912 checks per color.
                    # 4 * 6912 = 27648 checks per mapping.
                    # 10000 * 27648 = 276M checks. Might be okay in 110s?
                    # Python is slow. 276M simple ops might take 10-20s.
                    
                    # Let's optimize the inner loop.
                    # Precompute for each a, the list of b such that b+a is in mask?
                    # No, mask changes.
                    
                    # Let's just do it.
                    found = False
                    for b in range(48):
                        if not (mask & (1 << b)):
                            continue
                        if b == 0:
                            continue
                        s = add_table[a, b]
                        if s == 0:
                            continue
                        if mask & (1 << s):
                            found = True
                            break
                    if found:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Found a valid mapping!
                # Construct the coloring
                colors = []
                for i in range(48):
                    for j in range(i + 1, 48):
                        d, ds = edge_types[(i, j)]
                        colors.append(mapping[(d, ds)])
                
                # Verify
                if is_valid_coloring(48, colors):
                    return 48, colors
                else:
                    # Should not happen
                    continue
                    
        return None

    result_48 = try_group_construction_48()
    if result_48:
        n, colors = result_48
        return {'n': n, 'colors': colors}

    # Fallback to 32
    def mul16(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r

    log16, x = {}, 1
    for e in range(15):
        log16[x] = e
        x = mul16(x, 2)

    base16 = {}
    for i in range(16):
        for j in range(i + 1, 16):
            base16[(i, j)] = log16[i ^ j] % 3

    def get_color_32(i, j):
        if i // 16 == j // 16:
            return base16[(i % 16, j % 16)]
        else:
            return 3

    n32 = 32
    colors_32 = []
    for i in range(n32):
        for j in range(i + 1, n32):
            colors_32.append(get_color_32(i, j))
    
    if is_valid_coloring(n32, colors_32):
        return {'n': n32, 'colors': colors_32}
        
    return {'n': n32, 'colors': colors_32}

if __name__ == "__main__":
    res = construct()
    print(f"n={res['n']}, valid={is_valid_coloring(res['n'], res['colors'])}")
