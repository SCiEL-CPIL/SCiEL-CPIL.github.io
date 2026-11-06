import math

def construct():
    answers = []
    Z_LIMIT = 10**6  # Safe upper bound for z, solutions are usually small
    
    for n in range(3, 257):
        found = None
        # z must be > 3n/4
        start_z = (3 * n) // 4 + 1
        
        for z in range(start_z, Z_LIMIT + 1):
            # 1/x + 1/y = (4z - n) / (nz)
            # Let C = 4z - n
            # Cxy = nz(x+y)
            # x(Cy - nz) = nzy
            # y = nxz / (Cx - nz)
            # We need Cx - nz > 0 => x > nz / C
            # We need x < y < z
            # x < y => x < nxz / (Cx - nz) => Cx - nz < nx => x(C-n) < nz => x < nz / (C-n) = nz / (4z - 2n)
            # Wait, my previous derivation was:
            # 1/x + 1/y = C / (nz)
            # 1/x < C / (nz) => x > nz / C
            # 1/y < C / (nz) => y > nz / C
            # Since x < y, x is the smaller one.
            # Also 1/x > 1/y => x < y.
            # 1/x + 1/y > 2/x => C/(nz) > 2/x => x > 2nz/C.
            # So x > max(nz/C, 2nz/C) = 2nz/C.
            # And x < y => 1/x > 1/y => 1/x > C/(nz) - 1/x => 2/x > C/(nz) => x < 2nz/C.
            # This contradicts x > 2nz/C.
            # Let's re-evaluate.
            # 1/x + 1/y = K.
            # x < y implies 1/x > 1/y, so 1/x > K/2.
            # So x < 2/K.
            # Also 1/x < K, so x > 1/K.
            # So x is in (1/K, 2/K).
            # Here K = C / (nz) = (4z-n) / (nz).
            # So 1/K = nz / (4z-n).
            # 2/K = 2nz / (4z-n).
            # So x is in (nz / (4z-n), 2nz / (4z-n)).
            # This matches my earlier L and R.
            
            denom1 = 4 * z - n
            if denom1 <= 0:
                continue
            
            # L = floor(nz / denom1) + 1
            # R = floor(2nz / denom1)
            
            L = (n * z) // denom1 + 1
            R = (2 * n * z) // denom1
            
            if R >= z:
                R = z - 1  # x < z
            
            if L > R:
                continue
            
            for x in range(L, R + 1):
                num = x * denom1 - n * z
                if num <= 0:
                    continue
                if (n * x * z) % num != 0:
                    continue
                y = (n * x * z) // num
                if x < y < z:
                    found = [x, y, z]
                    break
            
            if found:
                break
        
        answers.append(found)
    return answers
