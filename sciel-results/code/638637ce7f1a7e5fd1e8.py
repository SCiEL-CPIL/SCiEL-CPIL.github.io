"""Square-product extension for 210: find distinct ints in (210, 10210]
whose product * 210 is a perfect square, minimizing max - 210.

Strategy:
1. Factorize all numbers in (210, 10210].
2. Represent each number by its square-free kernel as a vector over primes.
   Since we only care about parity, this is a vector in F_2.
   However, the number of primes up to 10210 is ~1200, so we can use integers as bitsets.
   Python integers have arbitrary precision, so we can map each prime to a bit position.
3. We want a subset S of {211, ..., M} such that XOR_{n in S} kernel(n) = kernel(210).
   Let K_210 be the bitset for 210.
   Let V_i be the bitset for number i.
   We want sum(V_i) mod 2 = K_210.
4. We can solve this using Gaussian elimination. We add numbers one by one (increasing M)
   to a basis. Once K_210 is in the span of the basis, we can reconstruct the subset.
   To minimize M, we should check after each addition if K_210 is now reachable.
   Actually, a more efficient way:
   - Precompute all bitsets.
   - Use a "linear algebra" approach to find the minimal M.
   - Note: The problem asks for 1..100 distinct integers. Gaussian elimination might produce a subset larger than 100.
   - However, if the dimension of the space spanned by numbers <= M is large enough, we can often find a small solution.
   - But wait, the constraint is "1..100 distinct integers".
   - If the Gaussian elimination solution has > 100 elements, it's invalid.
   - Is it possible that the minimal M requires > 100 elements?
   - Let's check the size of solutions.
   - Usually, if a vector is in the span, there are many solutions. We can try to minimize the size of the solution (weight minimization over F_2), which is NP-hard (closest vector problem), but for small dimensions or structured bases, we can do better.
   - Alternative: Since we want to minimize M, and M is likely small (around 210-300?), the number of available variables is small.
   - If M is small (e.g., 300), there are only ~90 variables. We can use BFS or DP over the state space? No, state space is 2^90.
   - Let's stick to Gaussian Elimination but ensure we can find a solution of size <= 100.
   - Actually, if we just need ANY solution, and the space is large, solutions are dense.
   - Let's try to find a solution of size <= 100.
   - If Gaussian elimination gives a solution of size > 100, we can try to reduce it by adding other linear dependencies (cycles in the cycle space of the matroid).
   - However, for the purpose of this task, let's first find the minimal M where a solution exists.
   - Then, we can try to minimize the size of the solution.
   - But the objective is ONLY to minimize M. The size constraint is just a hard bound (100).
   - So, if a solution of size <= 100 exists for M, we are good.
   - If the minimal M solution has size > 100, we might need a larger M to find a "sparser" solution?
   - This is tricky. Let's assume that for small M, the solutions are not too large.
   - 210 = 2*3*5*7.
   - If we pick 211 (prime), kernel is 211.
   - If we pick 212 = 4*53, kernel is 53.
   - If we pick 213 = 3*71, kernel is 3*71.
   - ...
   - Let's implement Gaussian Elimination to find the minimal M such that K_210 is in the span.
   - Then, reconstruct the solution. If size > 100, we might need to search for a different solution or increase M.
   - Given the density of square-free kernels, it's highly likely that a small solution exists.
   
   Implementation details:
   - Map primes to bit indices.
   - Compute bitset for each n in (210, 10210].
   - Maintain a basis of bitsets.
   - For each new n (in increasing order), try to add it to the basis.
   - If it adds a new dimension, add it.
   - After each addition (or even if it doesn't add a dimension, but we need to check reachability), check if K_210 is in the span.
   - To check reachability and reconstruct:
     - Keep track of the "reduced" form of the basis and the original vectors used.
     - Standard GE:
       - basis[bit] = (vector, original_index_list)
       - When adding a new vector v with index i:
         - v = v ^ basis[leading_bit] ...
         - If v becomes 0, it's dependent. We can update the representation of K_210 if we track it?
         - Better: Just maintain the basis. To check if K_210 is in span, try to reduce K_210 using the basis. If it reduces to 0, it's in the span.
         - To reconstruct the subset:
           - During reduction of K_210, keep track of which basis vectors were XORed.
           - Then, for each basis vector used, XOR in its associated "original set" of indices.
           - The final set of indices is the solution.
           
   - Wait, if a vector is dependent, we can still use it to form a different solution.
   - But for minimal M, we just need to know IF a solution exists.
   - The first M where K_210 reduces to 0 is the minimal M.
   - At that point, we reconstruct the solution.
   - If the solution size > 100, we have a problem.
   - However, note that if a vector is dependent, we can add it to the solution and XOR it with the current solution to get another solution.
   - Specifically, if v is dependent, v = sum(basis_vecs). Then v ^ sum(basis_vecs) = 0.
   - This doesn't help directly.
   - But if we have a solution S, and we have a dependent vector d (which is a sum of some basis vectors, and thus a sum of some original numbers), then S ^ support(d) is also a solution.
   - We can try to minimize the size of S by toggling dependent vectors.
   - But let's first see what size the GE solution has.
   
   Let's code the GE approach.
"""

import math
import time

def _is_square(n):
    r = math.isqrt(n)
    return r * r == n

def _sieve_primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def _factorize(n, primes):
    d = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            d[p] = e
    if n > 1:
        d[n] = 1
    return d

def construct():
    t_start = time.time()
    DEADLINE = t_start + 85.0 # Safety margin for 90s limit
    
    N = 210
    MAX_N = 10210
    MAX_K = 100
    
    primes = _sieve_primes(MAX_N)
    
    # Map prime to bit index
    # We only need primes that appear in numbers <= MAX_N
    # Actually, all primes <= MAX_N can appear.
    # Number of primes <= 10210 is approx 10210 / ln(10210) ~ 1000.
    # So bitsets will be ~1000 bits. Python handles this fine.
    prime_to_bit = {p: i for i, p in enumerate(primes)}
    
    # Precompute bitsets for all n in (210, MAX_N]
    # bitset[n] is an integer representing the square-free kernel of n
    bitsets = {}
    for n in range(211, MAX_N + 1):
        d = _factorize(n, primes)
        mask = 0
        for p, e in d.items():
            if e & 1:
                mask |= (1 << prime_to_bit[p])
        bitsets[n] = mask
        
    target = bitsets[N] # Kernel of 210. 210 = 2*3*5*7, all exp 1.
    
    # Gaussian Elimination
    # basis[bit] = (vector, frozenset_of_indices)
    # We want to find minimal M such that target is in span of {bitsets[i] for 211 <= i <= M}
    
    basis = {}
    
    # To reconstruct the solution, we need to know which original indices contributed to each basis vector.
    # When we add a new vector v with index i:
    #   v_reduced = v
    #   indices = {i}
    #   while v_reduced != 0:
    #       lb = v_reduced.bit_length() - 1
    #       if lb in basis:
    #           b_vec, b_indices = basis[lb]
    #           v_reduced ^= b_vec
    #           indices = indices ^ b_indices # Symmetric difference
    #       else:
    #           basis[lb] = (v_reduced, indices)
    #           break
    #   If v_reduced == 0, it's dependent. We can store the dependency relation if we want to minimize weight later.
    #   For now, we just check if target is in span.
    
    # We iterate M from 211 upwards.
    # At each step, we add bitsets[M] to the system.
    # Then we check if target is in the span.
    
    # To check if target is in span, we reduce target using the current basis.
    # If it reduces to 0, we have a solution.
    # The solution is the set of indices accumulated during reduction.
    
    best_solution = None
    best_M = None
    
    for M in range(211, MAX_N + 1):
        if time.time() > DEADLINE:
            break
            
        v = bitsets[M]
        indices = {M}
        v_reduced = v
        
        # Reduce v using existing basis
        while v_reduced != 0:
            lb = v_reduced.bit_length() - 1
            if lb in basis:
                b_vec, b_indices = basis[lb]
                v_reduced ^= b_vec
                indices = indices.symmetric_difference(b_indices)
            else:
                # Add to basis
                basis[lb] = (v_reduced, indices)
                break
        
        # Check if target is in span
        # Reduce target using basis
        t_reduced = target
        t_indices = set()
        
        while t_reduced != 0:
            lb = t_reduced.bit_length() - 1
            if lb in basis:
                b_vec, b_indices = basis[lb]
                t_reduced ^= b_vec
                t_indices = t_indices.symmetric_difference(b_indices)
            else:
                # Target not in span
                t_reduced = -1 # Flag
                break
        
        if t_reduced == 0:
            # Solution found!
            # t_indices is the set of indices that XOR to target.
            # Wait, the logic for t_indices:
            # We want sum(x_i * v_i) = target.
            # In GE, if target reduces to 0, it means target = sum(used_basis_vectors).
            # Each basis vector is a sum of original vectors.
            # So the set of original vectors is the union (with parity) of the indices in the used basis vectors.
            # My code does: t_indices = symmetric_difference(b_indices).
            # This is correct for XOR of sets.
            
            # Check size
            if len(t_indices) <= MAX_K:
                best_solution = sorted(list(t_indices))
                best_M = M
                break
            else:
                # Solution exists but size > 100.
                # We might need to find a smaller solution.
                # This is the "weight minimization" problem.
                # For now, let's note that a solution exists at M.
                # We can try to reduce the weight by using dependent vectors.
                # But let's first see if we can find a small M with small weight.
                # If not, we might have to accept a larger M or try to optimize weight.
                # Let's try to optimize weight for this M.
                
                # We have a solution S = t_indices.
                # We can add any linear dependency (cycle) to S to get another solution.
                # A dependency is a set of indices that XOR to 0.
                # During GE, when a vector v reduced to 0, we found a dependency:
                # v = sum(basis_vecs). So v ^ sum(basis_vecs) = 0.
                # The set of indices involved is {index_of_v} U union(basis_indices).
                # We can store these dependencies.
                
                # Let's restart the GE for this M but store dependencies?
                # Or just try random subsets of the current solution to see if we can reduce size?
                # No, that's not guaranteed.
                
                # Let's assume for now that if the first solution found has size > 100, 
                # we might need to search further or use a more advanced method.
                # However, given the structure, it's likely that a small solution exists for small M.
                # Let's continue searching for a solution with size <= 100.
                # If we find one, we break.
                # If we reach MAX_N without finding one, we return the best one found (even if > 100? No, invalid).
                # But the task says "1..100 distinct integers". So > 100 is invalid.
                
                # Let's try to find a smaller solution for this M.
                # We can use the fact that the null space (kernel of the matrix) contains dependencies.
                # If we have a basis for the row space, we can find a basis for the null space.
                # Then we can add null space vectors to S to minimize weight.
                # This is the syndrome decoding problem, which is hard.
                # But for small M (e.g., < 100), the number of variables is small.
                # We can use brute force / meet-in-the-middle?
                # If M is small, say 300, we have 89 variables.
                # We can split into two halves, compute all XOR sums for each half, and find a match.
                # But we need the sum to be exactly target.
                # This is possible but might be slow if M is large.
                
                # For now, let's just record that a solution exists but is too large, and continue.
                # If we find a later M with a smaller solution, we take it.
                # But we want MINIMAL M.
                # So if M is the minimal M where a solution exists, but it's too large, 
                # we might be stuck?
                # Actually, if a solution exists at M, it exists at any M' > M (by adding 0s? No, distinct integers).
                # Wait, we can't add 0s. We can only use distinct integers from (210, M'].
                # So the set of available vectors grows.
                # The span grows.
                # So if a solution exists at M, it exists at M+1?
                # Not necessarily the SAME solution, but a solution exists in the larger span.
                # So we can always find a solution for M >= M_min.
                # The question is whether we can find one with size <= 100.
                # As M increases, the null space dimension increases, giving more freedom to reduce weight.
                # So it's likely that a slightly larger M will have a smaller solution.
                
                # Let's continue the loop.
                pass
                
    if best_solution is not None:
        return best_solution
        
    # Fallback: If we didn't find a solution with size <= 100 in the deadline,
    # return the best solution found (even if > 100)? No, that's invalid.
    # We must return a valid list.
    # Let's try to find ANY solution, even if large, and then try to reduce it?
    # Or just return a known valid solution?
    # 840 = 4 * 210. Kernel(840) = 1. 210 * 840 = 210^2 * 4. Square.
    # So [840] is a valid solution of size 1.
    # Max(840) - 210 = 630.
    # This is a valid fallback.
    
    return [840]

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Max - 210: {max(result) - 210}")
    prod = 210
    for x in result:
        prod *= x
    print(f"Product is square: {_is_square(prod)}")
    print(f"Time: {t1 - t0:.2f}s")
