"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd, isqrt

DEADLINE = 85.0  # safety margin before 90s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _largest_prime_power_leq(p, limit):
    """Return the largest power of prime p that is <= limit."""
    result = 1
    while result * p <= limit:
        result *= p
    return result

def _initial_solution():
    """Construct initial solution: for each prime p<=256, use max(p, p^k) where p^k<=256, plus 1."""
    n = 256
    solution = [1]
    for p in range(2, n + 1):
        if _is_prime(p):
            pp = _largest_prime_power_leq(p, n)
            solution.append(pp)
    return solution

def _sum_of_solution(sol):
    return sum(sol)

def _is_valid_pairwise_coprime(sol):
    """Check if all pairs in sol are coprime."""
    for i in range(len(sol)):
        for j in range(i + 1, len(sol)):
            if gcd(sol[i], sol[j]) != 1:
                return False
    return True

def _can_add(sol, x, sol_set):
    """Check if x can be added to sol (coprime to all existing)."""
    if x in sol_set:
        return False
    for y in sol:
        if gcd(x, y) != 1:
            return False
    return True

def _local_search(sol, deadline):
    """Local search with 1-for-1, 1-for-2, 2-for-2 moves."""
    sol = list(sol)
    sol_set = set(sol)
    best_sum = sum(sol)
    n = 256
    all_nums = list(range(1, n + 1))
    
    # Precompute gcd compatibility
    # For speed, we'll check on the fly
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        sol_set = set(sol)
        
        # 1-for-1 moves: remove a, add b where b not in sol, gcd(b, all sol\{a}) == 1, and b > a
        for i, a in enumerate(sol):
            remaining = sol[:i] + sol[i+1:]
            rem_set = set(remaining)
            for b in all_nums:
                if b in sol_set:
                    continue
                if b <= a:
                    continue
                # Check b coprime to all in remaining
                ok = True
                for y in remaining:
                    if gcd(b, y) != 1:
                        ok = False
                        break
                if ok:
                    new_sol = remaining + [b]
                    new_sum = best_sum - a + b
                    if new_sum > best_sum:
                        sol = new_sol
                        best_sum = new_sum
                        improved = True
                        break
            if improved:
                break
        
        if improved or time.time() >= deadline:
            continue
        
        # 1-for-2 moves: remove a, add b and c
        sol_set = set(sol)
        for i, a in enumerate(sol):
            remaining = sol[:i] + sol[i+1:]
            rem_set = set(remaining)
            # Try adding pairs
            for bi in range(len(all_nums)):
                b = all_nums[bi]
                if b in sol_set:
                    continue
                # Check b coprime to remaining
                ok_b = True
                for y in remaining:
                    if gcd(b, y) != 1:
                        ok_b = False
                        break
                if not ok_b:
                    continue
                for ci in range(bi + 1, len(all_nums)):
                    c = all_nums[ci]
                    if c in sol_set or c == b:
                        continue
                    # Check c coprime to remaining and to b
                    ok_c = True
                    if gcd(b, c) != 1:
                        ok_c = False
                    else:
                        for y in remaining:
                            if gcd(c, y) != 1:
                                ok_c = False
                                break
                    if ok_c:
                        new_sum = best_sum - a + b + c
                        if new_sum > best_sum:
                            sol = remaining + [b, c]
                            best_sum = new_sum
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved or time.time() >= deadline:
            continue
        
        # 2-for-2 moves: remove a,b, add c,d
        sol_set = set(sol)
        for i in range(len(sol)):
            for j in range(i + 1, len(sol)):
                a, b = sol[i], sol[j]
                remaining = sol[:i] + sol[i+1:j] + sol[j+1:]
                rem_set = set(remaining)
                for ci in range(len(all_nums)):
                    c = all_nums[ci]
                    if c in sol_set:
                        continue
                    ok_c = True
                    for y in remaining:
                        if gcd(c, y) != 1:
                            ok_c = False
                            break
                    if not ok_c:
                        continue
                    for di in range(ci + 1, len(all_nums)):
                        d = all_nums[di]
                        if d in sol_set or d == c:
                            continue
                        ok_d = True
                        if gcd(c, d) != 1:
                            ok_d = False
                        else:
                            for y in remaining:
                                if gcd(d, y) != 1:
                                    ok_d = False
                                    break
                        if ok_d:
                            new_sum = best_sum - a - b + c + d
                            if new_sum > best_sum:
                                sol = remaining + [c, d]
                                best_sum = new_sum
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    
    return sol

def construct():
    start_time = time.time()
    
    # Initial solution
    sol = _initial_solution()
    
    # Verify validity
    if not _is_valid_pairwise_coprime(sol):
        # Fallback to primes + 1
        sol = [1] + [p for p in range(2, 257) if _is_prime(p)]
    
    # Local search
    sol = _local_search(sol, DEADLINE)
    
    # Final verification
    if not _is_valid_pairwise_coprime(sol):
        sol = [1] + [p for p in range(2, 257) if _is_prime(p)]
    
    # Ensure all are Python ints
    return [int(x) for x in sol]
# EVOLVE-BLOCK-END
