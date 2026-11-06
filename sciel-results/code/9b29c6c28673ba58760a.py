import time

def construct():
    # Generators for M12
    g1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
    g2 = [2, 0, 5, 1, 3, 7, 4, 11, 6, 8, 9, 10]
    
    # Inverses
    g1_inv = [11, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # g2_inv: inverse of (0, 2, 5, 7, 11, 10, 9, 8, 6, 4, 3, 1)
    # 0<-1, 1<-3, 3<-4, 4<-6, 6<-8, 8<-9, 9<-10, 10<-11, 11<-7, 7<-5, 5<-2, 2<-0
    g2_inv = [1, 3, 4, 6, 8, 9, 10, 11, 7, 5, 2, 0]
    
    def compose(p1, p2):
        return tuple(p1[p2[i]] for i in range(12))
        
    def apply_block(perm, block):
        return tuple(sorted(perm[i] for i in block))
        
    base_block = (0, 1, 2, 3, 4, 5)
    identity = tuple(range(12))
    
    seen_blocks = set()
    blocks = []
    
    queue = [identity]
    seen_perms = {identity}
    
    # Precompute neighbors for identity? No, compute on the fly.
    # Generators to apply: g1, g1_inv, g2, g2_inv
    gens = [g1, g1_inv, g2, g2_inv]
    
    while queue:
        if len(blocks) >= 132:
            break
            
        perm = queue.pop(0)
        
        # Apply to base block
        b = apply_block(perm, base_block)
        if b not in seen_blocks:
            seen_blocks.add(b)
            blocks.append(list(b))
            
            if len(blocks) == 132:
                break
                
        # Expand
        for g in gens:
            # new_perm = g o perm
            new_perm = compose(g, perm)
            if new_perm not in seen_perms:
                seen_perms.add(new_perm)
                queue.append(new_perm)
                
    return blocks
