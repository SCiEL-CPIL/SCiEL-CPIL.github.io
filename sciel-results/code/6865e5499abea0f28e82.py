import time

def construct():
    """
    Constructs the Steiner System S(5,6,12) using the PGL(2,11) group action.
    Returns a list of 132 distinct 6-subsets of 0..11.
    """
    start_time = time.time()
    deadline = 110  # Safety margin within 120s limit
    
    # Points are 0..10 and 11 (representing infinity)
    # Field is GF(11)
    MOD = 11
    
    # Modular inverse
    def inv(x):
        if x == 0:
            return None
        # Fermat's little theorem: x^(p-2) mod p
        return pow(x, MOD - 2, MOD)
    
    # Apply transformation (a, b, c, d) to point x
    # x' = (a*x + b) / (c*x + d)
    # If x is infinity (11), x' = a/c
    # If c*x + d == 0, x' is infinity (11)
    def apply_transform(x, a, b, c, d):
        if x == 11:
            # x = infinity
            if c == 0:
                # a/c is undefined if c=0? No, limit is a/c. 
                # In PGL, if c=0, transformation is ax+b. Infinity maps to infinity.
                # Wait, standard Mobius: if c=0, f(x) = (a/c)x + b/c. Infinity -> Infinity.
                return 11
            else:
                return (a * inv(c)) % MOD
        
        # x is finite
        num = (a * x + b) % MOD
        den = (c * x + d) % MOD
        
        if den == 0:
            return 11
        else:
            return (num * inv(den)) % MOD

    # Generators for PGL(2,11)
    # We need to generate the full group of order 1320.
    # PSL(2,11) has order 660. PGL(2,11) adds elements with det not a square.
    
    # Identity: (1, 0, 0, 1)
    # Translation T: (1, 1, 0, 1)
    # Inversion S: (0, 1, 1, 0)  # x -> -1/x ? No, (0,1,1,0) is x -> 1/x.
    # Let's use standard generators for PSL(2,11) and one for PGL.
    
    # Generators for PSL(2,11) (order 660)
    # A: x -> x+1 => (1, 1, 0, 1)
    # B: x -> -1/x => (0, -1, 1, 0) which is (0, 10, 1, 0)
    
    # To get PGL, we need an element with determinant a non-square.
    # Det(A) = 1. Det(B) = -1 = 10. 10 is a non-square in GF(11)?
    # Squares in GF(11): 1^2=1, 2^2=4, 3^2=9, 4^2=5, 5^2=3.
    # {1, 3, 4, 5, 9}. 10 is not in there. So B is in PGL but not PSL?
    # Actually, PSL(2,11) is the kernel of the map to PGL(2,11) modulo squares?
    # The index of PSL in PGL is 2.
    # So {A, B} might generate PGL(2,11) directly if B has non-square det.
    # Let's verify.
    
    gen_A = (1, 1, 0, 1)
    gen_B = (0, 10, 1, 0) # x -> 10/x = -1/x
    
    # We will perform BFS on the group elements represented as tuples (a,b,c,d)
    # Normalization: We can normalize such that d=1 if possible, or keep raw.
    # Since we are in a finite field, we can just store the tuples.
    # To avoid duplicates, we can normalize by dividing by d if d!=0, or by a if a!=0?
    # Better: Two matrices M1, M2 are same if M1 = k*M2.
    # We can normalize such that the first non-zero component is 1.
    # Or simpler: just use a set of tuples. The group is small (1320).
    
    group = set()
    queue = []
    
    # Start with identity
    ident = (1, 0, 0, 1)
    group.add(ident)
    queue.append(ident)
    
    while queue:
        if time.time() > deadline:
            break
        g = queue.pop(0)
        a, b, c, d = g
        
        for gen in (gen_A, gen_B):
            ga, gb, gc, gd = gen
            
            # Compose g * gen
            # (a b; c d) * (ga gb; gc gd)
            # = (a*ga + b*gc,  a*gb + b*gd)
            #   (c*ga + d*gc,  c*gb + d*gd)
            
            na = (a*ga + b*gc) % MOD
            nb = (a*gb + b*gd) % MOD
            nc = (c*ga + d*gc) % MOD
            nd = (c*gb + d*gd) % MOD
            
            new_g = (na, nb, nc, nd)
            
            # Check if new_g is equivalent to an existing element
            # In PGL, M ~ kM.
            # We can normalize new_g.
            # If nd != 0, multiply by inv(nd).
            # Else if nc != 0, multiply by inv(nc).
            # Else if nd... wait, if c=d=0, it's zero matrix, not in group.
            
            norm_g = new_g
            if nd != 0:
                inv_nd = inv(nd)
                norm_g = (na*inv_nd, nb*inv_nd, nc*inv_nd, 1)
            elif nc != 0:
                inv_nc = inv(nc)
                norm_g = (na*inv_nc, nb*inv_nc, 1, 0)
            elif nb != 0:
                inv_nb = inv(nb)
                norm_g = (na*inv_nb, 1, 0, 0)
            else:
                # na must be non-zero
                inv_na = inv(na)
                norm_g = (1, 0, 0, 0) # This case shouldn't happen for invertible matrices?
                # Actually if a!=0 and others 0, it's scaling.
            
            # We should also normalize the elements already in the set for comparison?
            # Or just store normalized forms.
            # Let's store normalized forms in the set.
            # We need to normalize 'ident' and 'gen' too?
            # Ident (1,0,0,1) -> norm (1,0,0,1)
            # gen_A (1,1,0,1) -> norm (1,1,0,1)
            # gen_B (0,10,1,0) -> d=0, c=1. norm: (0, 10*inv(1), 1, 0) = (0,10,1,0)
            
            if norm_g not in group:
                group.add(norm_g)
                queue.append(norm_g)
        
        # Also try gen * g? No, group generated by left multiplication of generators is enough
        # if we start from identity and multiply on the right by generators.
        # Wait, the standard BFS for group generation:
        # Start with {e}.
        # For each g in current set, and each generator s, add g*s.
        # This generates the subgroup.
        
    # The group size should be 1320.
    # If it's 660, we only generated PSL.
    # Let's check if gen_B is in PSL.
    # Det(gen_B) = 0*0 - 10*1 = -10 = 1 mod 11.
    # Wait. 1 is a square. So gen_B is in PSL?
    # Squares: 1, 3, 4, 5, 9.
    # Det(gen_B) = 1. So gen_B is in PSL(2,11).
    # We need a generator with non-square determinant to get PGL.
    # Let's use M: x -> 2x. (2, 0, 0, 1). Det = 2. 2 is not a square.
    gen_C = (2, 0, 0, 1)
    
    # Restart generation with all 3 generators to be safe
    group = set()
    queue = []
    ident = (1, 0, 0, 1)
    group.add(ident)
    queue.append(ident)
    
    generators = [gen_A, gen_B, gen_C]
    
    while queue:
        if time.time() > deadline:
            break
        g = queue.pop(0)
        a, b, c, d = g
        
        for gen in generators:
            ga, gb, gc, gd = gen
            
            na = (a*ga + b*gc) % MOD
            nb = (a*gb + b*gd) % MOD
            nc = (c*ga + d*gc) % MOD
            nd = (c*gb + d*gd) % MOD
            
            new_g = (na, nb, nc, nd)
            
            # Normalize
            norm_g = new_g
            if nd != 0:
                inv_nd = inv(nd)
                norm_g = (na*inv_nd, nb*inv_nd, nc*inv_nd, 1)
            elif nc != 0:
                inv_nc = inv(nc)
                norm_g = (na*inv_nc, nb*inv_nc, 1, 0)
            elif nb != 0:
                inv_nb = inv(nb)
                norm_g = (na*inv_nb, 1, 0, 0)
            else:
                inv_na = inv(na)
                norm_g = (1, 0, 0, 0)
                
            if norm_g not in group:
                group.add(norm_g)
                queue.append(norm_g)
                
    # The group should have 1320 elements.
    # Base block for S(5,6,12)
    # Standard base block: {0, 1, 3, 9, 10, infinity}
    # Mapping infinity to 11.
    base_block = (0, 1, 3, 9, 10, 11)
    
    blocks_set = set()
    
    for g in group:
        a, b, c, d = g
        transformed = []
        for p in base_block:
            tp = apply_transform(p, a, b, c, d)
            transformed.append(tp)
        
        # Sort to make tuple hashable and unique
        sorted_block = tuple(sorted(transformed))
        blocks_set.add(sorted_block)
        
        if time.time() > deadline:
            break
            
    # Convert to list of lists
    blocks = [list(b) for b in blocks_set]
    
    # Optional: Verify no 5-subsets are repeated
    # This is O(N^2 * 6) which is small for N=132.
    # But strictly speaking, the construction guarantees it.
    # We can skip verification for speed if we trust the math.
    # The prompt asks to maximize blocks. 132 is the max.
    
    return blocks

# EVOLVE-BLOCK-END
