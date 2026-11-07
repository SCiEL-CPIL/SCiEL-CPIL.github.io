import time
import math
import sys

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s
    n = 210
    max_val = 10210
    
    # Precompute square-free kernel for all numbers in range
    # kernel(x) = product of primes p where exponent of p in x is odd
    def compute_kernels(limit):
        kernels = [0] * (limit + 1)
        kernels[1] = 1
        # Sieve-like approach
        # For each prime p, for each multiple of p^2, divide out p^2
        # Actually, easier: factorize each number or use a modified sieve.
        # Let's use a simple sieve to compute the square-free part.
        
        # Initialize kernel[i] = i
        for i in range(1, limit + 1):
            kernels[i] = i
            
        # For each prime p, for each multiple of p^2, divide kernels[m] by p^2
        # This removes pairs of factors.
        p = 2
        while p * p <= limit:
            p2 = p * p
            if p2 > limit:
                break
            # For all multiples of p^2
            m = p2
            while m <= limit:
                # We need to divide kernels[m] by p^2 as many times as possible?
                # No, the kernel is defined by parity.
                # If x has p^e, and e is even, p is not in kernel. If e is odd, p is in kernel.
                # The sieve approach: start with kernel[i]=i.
                # For each prime p, for each multiple m of p^2, we "cancel" a pair of p's.
                # But we must do it carefully to handle higher powers.
                # Actually, a simpler way:
                # kernel(x) = x / (largest square dividing x)
                
                # Let's just factorize each number in the range 211..10210.
                # 10000 numbers is small.
                break
            p += 1
            
        # Recompute properly by factorization for clarity and correctness, 
        # as the sieve above was incomplete.
        def get_squarefree_kernel(x):
            kernel = 1
            d = 2
            while d * d <= x:
                if x % d == 0:
                    count = 0
                    while x % d == 0:
                        x //= d
                        count += 1
                    if count % 2 == 1:
                        kernel *= d
                d += 1
            if x > 1:
                kernel *= x
            return kernel

        for i in range(211, limit + 1):
            kernels[i] = get_squarefree_kernel(i)
            
        return kernels

    # Target: product of selected integers * 210 is a perfect square.
    # Let S be the set of selected integers.
    # kernel(210 * prod(S)) = 1.
    # kernel(210 * prod(S)) = kernel(210) * prod(kernel(s) for s in S) mod squares.
    # Since kernels are square-free, this means the multiset of prime factors 
    # of kernel(210) and all kernel(s) must have even counts.
    # Equivalently, the XOR of the parity vectors of the prime factors is 0.
    # Or simpler: kernel(210) * prod(kernel(s)) must be a perfect square.
    # Since they are all square-free, their product is a perfect square iff 
    # the set of prime factors is a multiset where each prime appears an even number of times.
    # This is equivalent to: the product of the kernels is a perfect square.
    # Wait, kernel(210) is 210 (since 210 = 2*3*5*7, all exponents 1).
    # So we need kernel(210) * prod(kernel(s_i)) = k^2.
    # This implies prod(kernel(s_i)) = 210 * (k / kernel(210))^2 ? No.
    # Let K = kernel(210) = 210.
    # We need K * K_1 * K_2 ... K_m to be a square.
    # This is equivalent to K_1 * K_2 ... K_m being of the form K * square.
    # Since K_i are square-free, this means the symmetric difference of prime sets is empty.
    # In other words, the product of the kernels must be equal to 210 * (some square).
    # But since the product of square-free numbers is square-free only if they are coprime? No.
    # Example: 2 * 2 = 4 (square). Kernel(2)=2, Kernel(2)=2. Product=4.
    # So we need the product of the kernels of the selected numbers to be a multiple of 210, 
    # such that when divided by 210, the result is a perfect square.
    # Actually, simpler condition: 
    # Let P = prod(s_i). We need 210 * P = square.
    # This is equivalent to kernel(210 * P) = 1.
    # kernel(210 * P) = kernel(210) * kernel(P) if 210 and P are "compatible"? No.
    # kernel(xy) is not simply kernel(x)kernel(y).
    # However, if we compute the prime factorization parity vector for 210 and each s_i,
    # we just need the sum of parity vectors (mod 2) to be zero.
    
    # Let's use the property: 
    # 210 * prod(s_i) is a square iff kernel(210 * prod(s_i)) == 1.
    # We can compute this directly for candidate sets.
    
    # Strategy:
    # 1. Find pairs (a, b) such that 210 * a * b is a square.
    #    This means kernel(210 * a * b) == 1.
    #    Let k_a = kernel(a), k_b = kernel(b).
    #    Is there a direct relation?
    #    210 * a * b = 210 * (k_a * sq_a) * (k_b * sq_b) = 210 * k_a * k_b * (sq_a * sq_b).
    #    Since sq_a*sq_b is a square, 210 * a * b is a square iff 210 * k_a * k_b is a square.
    #    So we need kernel(210 * k_a * k_b) == 1.
    #    Since k_a, k_b are square-free, let's just check this condition.
    
    # 2. We want to minimize max(a, b) - 210.
    #    So we iterate b from 211 upwards, and for each b, find the smallest a > 210, a != b,
    #    such that 210 * a * b is a square.
    #    Wait, a and b are distinct.
    #    If we fix b, we need a such that 210 * a * b is a square.
    #    This implies a = 210 * b * k^2 / (210 * b * k^2 / a)? No.
    #    210 * a * b = s^2 => a = s^2 / (210 * b).
    #    So a must be an integer.
    #    Also a > 210, a != b, a <= 10210.
    #    
    #    Let's precompute for each b in [211, 10210], what a's are possible?
    #    Or, iterate a from 211 to 10210, and b from a+1 to 10210? That's 50M pairs, too slow.
    #    
    #    Better: 
    #    For each b in [211, 10210]:
    #       We need a such that 210 * a * b is a square.
    #       Let g = gcd(210 * b, 210 * b)? No.
    #       Let M = 210 * b.
    #       We need a such that M * a is a square.
    #       This means a = M * k^2 / (square part of M)?
    #       Let M = s * u^2 where s is square-free.
    #       Then M * a = s * u^2 * a = square => s * a must be a square.
    #       Since s is square-free, a must be of the form s * v^2.
    #       So a = s * v^2.
    #       We need a > 210, a != b, a <= 10210.
    #       And a must be distinct from b.
    #       Also, we want to minimize max(a, b). Since we iterate b, if we find a valid a,
    #       the value is max(a, b) - 210.
    #       To minimize the global max, we should check small b's first?
    #       If we find a pair with max(a,b) = V, we can stop if we've checked all b < V?
    #       Not exactly, because a could be large.
    #       But if we iterate b from 211 upwards, and for each b find the smallest valid a,
    #       we can track the best max(a,b).
    #       
    #    Let's implement this.
    
    def get_squarefree_part(x):
        # Returns (s, u) such that x = s * u^2 and s is square-free.
        s = 1
        u = 1
        d = 2
        while d * d <= x:
            if x % d == 0:
                count = 0
                while x % d == 0:
                    x //= d
                    count += 1
                if count % 2 == 1:
                    s *= d
                u *= d ** (count // 2)
            d += 1
        if x > 1:
            s *= x
        return s, u

    # Precompute square-free part for all numbers in range?
    # We need it for M = 210 * b.
    # b goes up to 10210. M goes up to 210 * 10210 ~ 2.1M.
    # We can compute on the fly or precompute for b.
    
    best_value = 630  # From [840]
    best_result = [840]
    
    # Iterate b from 211 to 10210
    # For each b, compute s_b, u_b such that 210*b = s_b * u_b^2
    # Then a must be s_b * v^2.
    # We need a > 210, a != b, a <= 10210.
    # Also a must be integer.
    
    # Optimization: 
    # If we find a pair with max(a,b) = V, we can stop checking b if b > V 
    # and we know that any future pair will have max >= b > V.
    # So we can break if b > best_value + 210.
    
    for b in range(211, 10211):
        if b - 210 >= best_value:
            # If b itself is already too large, any pair involving b will have max >= b
            # So max(a,b) >= b > best_value + 210.
            # We can break.
            break
            
        M = 210 * b
        s, u = get_squarefree_part(M)
        
        # a = s * v^2
        # We need a > 210, a != b, a <= 10210.
        # v must be integer >= 1.
        # a_min = 211 (since a > 210)
        # s * v^2 >= 211 => v^2 >= 211/s => v >= ceil(sqrt(211/s))
        # s * v^2 <= 10210 => v^2 <= 10210/s => v <= floor(sqrt(10210/s))
        
        v_min = math.isqrt((211 + s - 1) // s)
        if v_min * v_min < 211 // s: # careful with integer division
             # Better: v_min = ceil(sqrt(211/s))
             # Let's just start from v=1 and check.
             pass
        
        # Let's just iterate v from 1 up to sqrt(10210/s)
        max_v = math.isqrt(10210 // s)
        
        for v in range(1, max_v + 1):
            a = s * v * v
            if a <= 210:
                continue
            if a == b:
                continue
            if a > 10210:
                break
            
            # Check if a is distinct from b (already done)
            # Check if a is integer (it is)
            # Check if 210 * a * b is a square (by construction it should be, but let's verify for safety)
            # Actually, the construction guarantees it.
            
            val = max(a, b) - 210
            if val < best_value:
                best_value = val
                best_result = [int(a), int(b)]
                
        # If best_value is very small, we might break early?
        # The loop break condition handles it.
        
    # Also check single element case: a = 840, value = 630.
    # Our loop might find better pairs.
    
    # Verify the best result
    def is_square(m):
        if m < 0:
            return False
        s = math.isqrt(m)
        return s * s == m

    prod = 210
    for x in best_result:
        prod *= x
    if not is_square(prod):
        # Fallback
        return [840]
        
    return best_result

if __name__ == "__main__":
    print(construct())
