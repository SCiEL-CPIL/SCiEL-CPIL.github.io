from math import gcd


def construct():
        # Select all even moduli in [20, 200]
        moduli = [m for m in range(20, 201) if m % 2 == 0]
        
        selected = []
        for m in moduli:
            valid_r = None
            for r in range(m):
                ok = True
                for m_j, r_j in selected:
                    g = gcd(m, m_j)
                    if r % g == r_j % g:
                        ok = False
                        break
                if ok:
                    valid_r = r
                    break
            
            if valid_r is not None:
                selected.append([m, valid_r])
        
        return selected
