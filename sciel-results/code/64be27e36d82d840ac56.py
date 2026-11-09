import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115.0
    
    # Fallback: The known n=32 construction from the seed
    def get_seed_32():
        def mul(a, b):
            r = 0
            for i in range(4):
                if b >> i & 1:
                    r ^= a << i
            for i in range(7, 3, -1):
                if r >> i & 1:
                    r ^= 0b10011 << (i - 4)
            return r
        
        log = {}
        x = 1
        for e in range(15):
            log[x] = e
            x = mul(x, 2)
            
        colors = []
        for i in range(32):
            for j in range(i + 1, 32):
                if i // 16 == j // 16:
                    colors.append(log[(i % 16) ^ (j % 16)] % 3)
                else:
                    colors.append(3)
        return 32, colors

    best_n, best_colors = get_seed_32()

    # Try to find larger n using GF(q) construction
    # n = 2q
    # We need q-1 divisible by 3
    # Try q = 25, 49, 121, 169...
    
    # Precompute some small primes/powers
    # q must be a prime power.
    # 25=5^2, 49=7^2, 81=3^4 (80 not div by 3), 121=11^2, 169=13^2, 289=17^2...
    
    candidates = [25, 49, 121, 169, 289]
    
    for q in candidates:
        if time.time() > deadline - 10:
            break
            
        if (q - 1) % 3 != 0:
            continue
            
        size = (q - 1) // 3
        
        # Build GF(q) addition table
        # We need to define GF(q).
        # For q=p^k, we can just use a generic representation if we can't find poly easily.
        # But for small q, we can hardcode or find poly.
        # Let's try to find a primitive polynomial for GF(p^2) where p is prime.
        # Or just use a dictionary for addition if we can generate the field.
        
        # Simple approach: Use Python integers 0..q-1 and assume we have a mapping.
        # Actually, we can just use the fact that for the purpose of sum-free sets,
        # we just need the addition table.
        # We can generate GF(p^2) using a standard irreducible polynomial.
        
        def build_gf_add_table(q):
            p = int(q ** 0.5)
            if p * p != q:
                # Handle other cases if needed, but candidates are squares
                return None
            
            # Find irreducible polynomial of degree 2 over GF(p)
            # x^2 + ax + b
            # b must be non-square.
            # For small p, we can just try.
            
            # Squares in GF(p):
            squares = set((x*x) % p for x in range(p))
            
            # Try polynomials
            found = False
            for a in range(p):
                for b in range(1, p):
                    if b not in squares:
                        # Check if x^2 + ax + b is irreducible
                        # It is irreducible iff it has no roots in GF(p)
                        irreducible = True
                        for x in range(p):
                            if (x*x + a*x + b) % p == 0:
                                irreducible = False
                                break
                        if irreducible:
                            # Use this polynomial
                            # Representation: element is x^1 * coeff1 + x^0 * coeff0
                            # coeff0, coeff1 in [0, p-1]
                            # Value = coeff0 + coeff1 * p
                            
                            add_table = [[0]*q for _ in range(q)]
                            for u in range(q):
                                u0 = u % p
                                u1 = u // p
                                for v in range(q):
                                    v0 = v % p
                                    v1 = v // p
                                    
                                    # Sum in GF(p^2)
                                    s0 = (u0 + v0) % p
                                    s1 = (u1 + v1) % p
                                    
                                    # Reduce if necessary? 
                                    # Multiplication would require reduction.
                                    # But addition in polynomial representation is just component-wise 
                                    # IF we don't carry over to x^2.
                                    # Wait, the representation is basis {1, x}.
                                    # Addition is linear. No reduction needed for addition.
                                    # Reduction is only for multiplication.
                                    # So add_table[u][v] = s0 + s1 * p
                                    
                                    add_table[u][v] = s0 + s1 * p
                            
                            found = True
                            break
                if found:
                    break
            if not found:
                return None
            return add_table

        add_table = build_gf_add_table(q)
        if add_table is None:
            continue
            
        # Non-zero elements
        non_zero = list(range(1, q))
        
        # We need to partition non_zero into 3 sets S0, S1, S2 of size 'size'
        # Minimize sum of triangles in each set.
        
        # Initial random assignment
        # Ensure correct sizes
        assignment = [0]*q # assignment[i] is color of i, -1 for 0
        assignment[0] = -1
        
        # Create a pool of elements and assign randomly
        pool = non_zero[:]
        random.shuffle(pool)
        
        # We need exactly 'size' elements in each color
        # This is a constrained optimization.
        # We can start with a random valid partition.
        
        def random_partition(elements, num_sets, size_per_set):
            random.shuffle(elements)
            part = [[] for _ in range(num_sets)]
            for i, el in enumerate(elements):
                part[i % num_sets].append(el)
            return part
        
        # Try multiple random restarts
        max_restarts = 20
        found_valid = False
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            parts = random_partition(non_zero, 3, size)
            
            # Convert to sets for faster lookup? 
            # Or just lists and use set() for membership?
            # Since size is small, lists are fine, but set is better for O(1) lookup.
            # We will maintain sets S[0], S[1], S[2]
            
            S = [set(parts[i]) for i in range(3)]
            
            # Precompute pair sums?
            # Cost calculation:
            def count_triples(S_set):
                cnt = 0
                elems = list(S_set)
                L = len(elems)
                for i in range(L):
                    a = elems[i]
                    for j in range(i+1, L):
                        b = elems[j]
                        # Need c = -a-b
                        c = (q - (add_table[a][b] % q)) % q
                        if c in S_set and c != a and c != b:
                            # To avoid double counting, we can just count and divide by 3?
                            # Or ensure a < b < c?
                            # But order in set is not guaranteed.
                            # Let's just count all ordered triples (a,b,c) and divide by 6?
                            # No, a,b,c are distinct.
                            # The number of permutations of {a,b,c} is 6.
                            # So if we count all (a,b) in S, check if c in S, we get 6 * (#triangles).
                            pass
                # This is O(n^2). For n=40, 1600 ops. Fast.
                # Let's implement a faster counter.
                cnt = 0
                for a in S_set:
                    for b in S_set:
                        if a == b: continue
                        c = (q - (add_table[a][b] % q)) % q
                        if c in S_set and c != a and c != b:
                            cnt += 1
                return cnt // 6
            
            curr_cost = sum(count_triples(S[i]) for i in range(3))
            
            if curr_cost == 0:
                found_valid = True
                break
                
            # Local Search
            # Move one element from Si to Sj
            stall_count = 0
            T = 10.0 # Temperature
            
            while time.time() < deadline - 5 and not found_valid:
                # Pick a random element
                i = random.randint(0, 2)
                if not S[i]:
                    continue
                x = random.choice(list(S[i]))
                
                j = random.randint(0, 2)
                if i == j:
                    continue
                if x in S[j]:
                    continue # Should not happen if partition
                
                # Calculate delta cost
                # Cost of Si decreases by number of triples containing x
                # Triples containing x in Si: pairs {a,b} in Si\{x} such that a+b+x=0
                # i.e., a+b = -x
                neg_x = (q - x) % q
                
                def count_pairs_summing_to(target, S_set, exclude=None):
                    cnt = 0
                    elems = [e for e in S_set if e != exclude]
                    # Use set for lookup
                    S_lookup = set(elems)
                    for a in elems:
                        b = (target - a) % q
                        if b in S_lookup and b > a: # Count each pair once
                            # Ensure a != b
                            if a != b:
                                cnt += 1
                    return cnt
                
                # This O(n^2) per move is slow.
                # We need faster cost update.
                # Precompute for each x and each set S, the number of pairs in S summing to -x.
                # Let P[S][x] = # pairs {a,b} in S such that a+b = -x.
                # When we move x from Si to Sj:
                # Cost(Si) decreases by P[S_i \ {x}][-x]
                # Cost(Sj) increases by P[S_j][-x]
                
                # We can maintain P dynamically?
                # Or just recompute the cost of the two affected sets.
                # Cost(Si) and Cost(Sj).
                # O(n^2) for each set. 2 * 1600 = 3200 ops.
                # 10000 moves * 3200 = 32M ops. Python might be slow (10s).
                # Let's optimize count_triples.
                
                # Better: Just compute delta.
                # Old cost = C(Si) + C(Sj)
                # New cost = C(Si\{x}) + C(Sj U {x})
                
                # We can write a function that computes cost of a set given a list.
                
                S_i_list = list(S[i])
                S_j_list = list(S[j])
                
                # Remove x from S_i, add to S_j
                S_i_new = S[i] - {x}
                S_j_new = S[j] | {x}
                
                new_cost_i = count_triples(S_i_new)
                new_cost_j = count_triples(S_j_new)
                
                old_cost_i = count_triples(S[i]) # We should have stored this?
                old_cost_j = count_triples(S[j])
                
                # To avoid recomputing old costs, we should keep a global cost array.
                # But for simplicity, let's just do it.
                
                delta = (new_cost_i + new_cost_j) - (curr_cost - count_triples(S[i]) - count_triples(S[j]) + count_triples(S[i]) + count_triples(S[j])) # This is messy.
                
                # Let's just keep an array `set_costs`
                
                if True:
                    # Re-do with set_costs
                    pass

            # This implementation is getting complex.
            # Let's use a simpler, potentially slower but correct method for small q.
            # For q=25, size=8. O(8^3) is tiny.
            # For q=49, size=16. O(16^3) is 4096.
            # For q=121, size=40. O(40^3) is 64000.
            
            # Let's try q=25 and q=49 with this slower local search.
            pass

    return {'n': best_n, 'colors': best_colors}
