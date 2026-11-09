import time

def construct():
    START = time.time()
    TIME_LIMIT = 140.0
    
    p = 11
    points = list(range(12))
    
    # Generate PGL(2, 11)
    group_elements = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    det = (a * d - b * c) % p
                    if det != 0:
                        group_elements.append((a, b, c, d))
                        
    def apply_transform(x, a, b, c, d):
        if x == 11: # infinity
            if c == 0:
                return 11
            else:
                return (a * pow(c, p-2, p)) % p
        else:
            denom = (c * x + d) % p
            if denom == 0:
                return 11
            else:
                num = (a * x + b) % p
                return (num * pow(denom, p-2, p)) % p

    # Base block candidate
    base_block = frozenset([0, 1, 3, 4, 9, 11])
    
    blocks = set()
    for (a, b, c, d) in group_elements:
        new_block = frozenset(apply_transform(x, a, b, c, d) for x in base_block)
        blocks.add(new_block)
        
        if time.time() - START > TIME_LIMIT:
            return [list(b) for b in blocks]
            
    result_blocks = [list(b) for b in blocks]
    
    return result_blocks
