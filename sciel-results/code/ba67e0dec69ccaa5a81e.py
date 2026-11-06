import time

def construct():
    # Construct S(5,6,12) using PSL(2,11)
    
    def mod_inv(a, m=11):
        for i in range(1, m):
            if (a * i) % m == 1:
                return i
        return 0

    group_elements = []
    seen = set()
    
    for a in range(11):
        for b in range(11):
            for c in range(11):
                for d in range(11):
                    det = (a * d - b * c) % 11
                    if det == 1:
                        na, nb, nc, nd = (-a)%11, (-b)%11, (-c)%11, (-d)%11
                        if (a,b,c,d) <= (na,nb,nc,nd):
                            if (a,b,c,d) not in seen:
                                seen.add((a,b,c,d))
                                group_elements.append((a,b,c,d))
                                
    def apply_m(matrix, point):
        a, b, c, d = matrix
        if point == 11:
            if c == 0:
                return 11
            else:
                return a * mod_inv(c, 11) % 11
        else:
            num = (a * point + b) % 11
            den = (c * point + d) % 11
            if den == 0:
                return 11
            else:
                return num * mod_inv(den, 11) % 11

    base_block = (0, 1, 2, 3, 4, 5)
    blocks = set()
    for m in group_elements:
        new_block = tuple(sorted(apply_m(m, p) for p in base_block))
        blocks.add(new_block)
        
    return list(blocks)
