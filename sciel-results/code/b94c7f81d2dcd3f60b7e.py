import time
from itertools import combinations

def construct():
    """
    Finds a 2-coloring of K_n^(3) with no Red K_4^(3) and no Blue K_5^(3).
    Uses Deterministic Greedy Construction with bitset optimization.
    """
    start_time = time.time()
    time_limit = 110  # Leave 10s for overhead
    
    # Helper to check if a specific coloring is valid (fallback)
    def is_valid(n, colors):
        # Reconstruct masks
        red_mask = [[0 for _ in range(n)] for _ in range(n)]
        blue_mask = [[0 for _ in range(n)] for _ in range(n)]
        
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    if colors[idx] == 0: # Red
                        red_mask[a][b] |= (1 << c)
                        red_mask[b][a] |= (1 << c)
                        red_mask[a][c] |= (1 << b)
                        red_mask[c][a] |= (1 << b)
                        red_mask[b][c] |= (1 << a)
                        red_mask[c][b] |= (1 << a)
                    else: # Blue
                        blue_mask[a][b] |= (1 << c)
                        blue_mask[b][a] |= (1 << c)
                        blue_mask[a][c] |= (1 << b)
                        blue_mask[c][a] |= (1 << b)
                        blue_mask[b][c] |= (1 << a)
                        blue_mask[c][b] |= (1 << a)
                    idx += 1
        
        # Check Red K4
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    candidates = red_mask[a][b] & red_mask[a][c] & red_mask[b][c]
                    candidates &= ~((1 << (c+1)) - 1)
                    if candidates:
                        return False
        
        # Check Blue K5
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    candidates = blue_mask[a][b] & blue_mask[a][c] & blue_mask[b][c]
                    candidates &= ~((1 << (c+1)) - 1)
                    cands = []
                    temp = candidates
                    while temp:
                        lsb = temp & -temp
                        v = lsb.bit_length() - 1
                        cands.append(v)
                        temp ^= lsb
                    for i in range(len(cands)):
                        d = cands[i]
                        for j in range(i+1, len(cands)):
                            e = cands[j]
                            if (blue_mask[a][d] & (1 << e)) and \
                               (blue_mask[b][d] & (1 << e)) and \
                               (blue_mask[c][d] & (1 << e)):
                                return False
        return True

    # Try different n from high to low
    for n in range(40, 8, -1):
        if time.time() - start_time > time_limit:
            break
            
        # Greedy Construction
        # colors[i] is the color of the i-th triple in lex order
        num_triples = n * (n-1) * (n-2) // 6
        colors = [-1] * num_triples
        
        # Precompute triple indices
        triple_indices = {}
        idx = 0
        triples_list = []
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triple_indices[(a, b, c)] = idx
                    triples_list.append((a, b, c))
                    idx += 1
        
        # Masks for checking violations
        # red_mask[u][v] has bit w set if (min(u,v), max(u,v), w) is Red?
        # Better: red_mask[a][b] stores bits of c such that (a,b,c) is Red.
        # Since a<b<c, we store in red_mask[a][b] bit c.
        # But for checking "is (a,c,d) red?", we need red_mask[a][c] & (1<<d).
        red_mask = [[0 for _ in range(n)] for _ in range(n)]
        blue_mask = [[0 for _ in range(n)] for _ in range(n)]
        
        success = True
        
        for idx, (a, b, c) in enumerate(triples_list):
            if time.time() - start_time > time_limit:
                success = False
                break
                
            # Check if coloring Red is safe
            # Red K4 violation if there exists d such that (a,b,c) is the LAST edge in {a,b,c,d}
            # and (a,b,d), (a,c,d), (b,c,d) are Red.
            
            # Which d makes (a,b,c) the last edge?
            # Lex order of {a,b,c,d} triples:
            # If d > c:
            #   (a,b,c) < (a,b,d) < (a,c,d) < (b,c,d)
            #   So (a,b,c) is FIRST. Not last.
            # If d < a:
            #   (d,a,b) < (d,a,c) < (d,b,c) < (a,b,c)
            #   So (a,b,c) is LAST.
            # If a < d < b:
            #   (a,d,b) ... wait, d is between a and b.
            #   Vertices: a, d, b, c.
            #   Triples: (a,d,b) -> (a,d,b) is not valid order, it's (a,d,b) with a<d<b.
            #   Sorted vertices: a, d, b, c.
            #   Triples: (a,d,b), (a,d,c), (a,b,c), (d,b,c).
            #   Order: (a,d,b) < (a,d,c) < (a,b,c) < (d,b,c).
            #   So (a,b,c) is 3rd. (d,b,c) is last.
            # If b < d < c:
            #   Vertices: a, b, d, c.
            #   Triples: (a,b,d), (a,b,c), (a,d,c), (b,d,c).
            #   Order: (a,b,d) < (a,b,c) < (a,d,c) < (b,d,c).
            #   So (a,b,c) is 2nd.
            
            # So (a,b,c) is the LAST edge in a K4 {a,b,c,d} ONLY IF d < a.
            # Let's verify.
            # Set S = {a,b,c,d}. Sorted: x1 < x2 < x3 < x4.
            # The last triple in lex order is (x2, x3, x4).
            # So (a,b,c) is the last triple iff {a,b,c} = {x2, x3, x4}.
            # This implies d = x1, so d < a.
            
            red_safe = True
            # Check all d < a
            for d in range(a):
                # Check if (a,b,d), (a,c,d), (b,c,d) are Red
                # Note: d < a < b < c
                # (a,b,d) is stored in red_mask[d][a] bit b? Or red_mask[a][d] bit b?
                # My mask definition: red_mask[u][v] has bit w if (u,v,w) is Red with u<v<w.
                # So for (d,a,b): u=d, v=a, w=b. Check red_mask[d][a] & (1<<b).
                # For (d,a,c): u=d, v=a, w=c. Check red_mask[d][a] & (1<<c).
                # For (d,b,c): u=d, v=b, w=c. Check red_mask[d][b] & (1<<c).
                
                if (red_mask[d][a] & (1 << b)) and \
                   (red_mask[d][a] & (1 << c)) and \
                   (red_mask[d][b] & (1 << c)):
                    red_safe = False
                    break
            
            # Check if coloring Blue is safe
            # Blue K5 violation if there exist d,e such that (a,b,c) is the LAST edge in {a,b,c,d,e}
            # and the other 9 are Blue.
            # (a,b,c) is last edge in K5 {x1..x5} iff {a,b,c} = {x3, x4, x5}.
            # So d, e must be < a.
            # Let d < e < a.
            # Vertices: d, e, a, b, c.
            # Last triple: (a,b,c).
            # Other 9 triples must be Blue.
            # We need to check all pairs (d,e) with d < e < a.
            
            blue_safe = True
            for e in range(1, a):
                for d in range(e):
                    # Check if all 9 other triples in {d,e,a,b,c} are Blue
                    # Triples:
                    # (d,e,a), (d,e,b), (d,e,c)
                    # (d,a,b), (d,a,c), (d,b,c)
                    # (e,a,b), (e,a,c), (e,b,c)
                    
                    if not (blue_mask[d][e] & (1 << a)): continue
                    if not (blue_mask[d][e] & (1 << b)): continue
                    if not (blue_mask[d][e] & (1 << c)): continue
                    if not (blue_mask[d][a] & (1 << b)): continue
                    if not (blue_mask[d][a] & (1 << c)): continue
                    if not (blue_mask[d][b] & (1 << c)): continue
                    if not (blue_mask[e][a] & (1 << b)): continue
                    if not (blue_mask[e][a] & (1 << c)): continue
                    if not (blue_mask[e][b] & (1 << c)): continue
                    
                    # If all 9 are Blue, then coloring (a,b,c) Blue creates a Blue K5
                    blue_safe = False
                    break
                if not blue_safe:
                    break
            
            if red_safe and blue_safe:
                # Pick Red (0)
                colors[idx] = 0
                red_mask[a][b] |= (1 << c)
                # Update symmetric masks for consistency? 
                # My check logic uses specific indices like red_mask[d][a].
                # I need to ensure red_mask[u][v] is populated for u<v.
                # I only populate red_mask[a][b] where a<b.
                # The check logic above:
                # red_mask[d][a] where d<a. Correct.
                # red_mask[d][b] where d<b. Correct.
                # So I only need to populate red_mask[min][max].
                pass
            elif red_safe:
                # Must pick Blue (1)
                colors[idx] = 1
                blue_mask[a][b] |= (1 << c)
            elif blue_safe:
                # Must pick Red (0)
                colors[idx] = 0
                red_mask[a][b] |= (1 << c)
            else:
                # Conflict
                success = False
                break
        
        if success:
            # Verify
            if is_valid(n, colors):
                return {'n': n, 'colors': colors}
            # If greedy produced invalid (shouldn't happen if logic is right), try next n?
            # Or just return it if is_valid fails, we might have a bug in is_valid or greedy.
            # Given the strictness, if greedy says valid, it should be.
            # But is_valid is the ground truth.
            # If greedy fails is_valid, we skip this n.

    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
