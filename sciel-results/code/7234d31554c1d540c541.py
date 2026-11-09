"""Rewritten program: CSP-style greedy with backtracking for prime residue covering."""
import sys
from fractions import Fraction
from math import isqrt

# Precompute primes up to 1000
def _primes_up_to(n):
    sieve = bytearray(b'\x01') * (n + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, n + 1) if sieve[i]]

PRIMES_1000 = _primes_up_to(1000)
# Map prime to index for quick lookup
PRIME_INDEX = {p: i for i, p in enumerate(PRIMES_1000)}

def construct():
    """
    Find a set of distinct primes p <= 1000 with residues r (0 <= r < p)
    such that sum(1/p) <= 1 and every integer 1..L is covered.
    Maximize L.
    """
    # Budget is 1 (exact rational)
    budget = Fraction(1, 1)
    
    # We want to cover 1..L for as large L as possible.
    # Strategy: Greedy with backtracking / local search.
    # 
    # Key insight: To maximize L, we need to cover consecutive integers starting from 1.
    # The seed covers 1..~900 using primes 13..~1000 with residue (i+1)%p.
    # 
    # A better approach: 
    # 1. Start with a greedy assignment that covers 1..L as much as possible.
    # 2. Use local search / hill climbing to improve coverage.
    #
    # Let's try a greedy approach first:
    # For each integer n from 1 upward, if not yet covered, assign a prime p and residue r
    # such that n ≡ r (mod p). We want to minimize the cost 1/p added to the budget.
    # But we also want to cover future integers efficiently.
    #
    # Actually, let's think differently. The classic covering systems use small primes.
    # But here we're limited to primes <= 1000 and budget sum(1/p) <= 1.
    #
    # Let's try: For each prime p in some order, assign a residue r to cover as many
    # currently-uncovered integers as possible, while staying within budget.
    #
    # Order primes by... let's try descending (large primes first) or ascending.
    # Large primes cover fewer integers per residue class in a given range but have small 1/p.
    # Small primes cover more integers per residue but have larger 1/p.
    #
    # Let's try a greedy: iterate n from 1 to L_max (say 100000), and for each uncovered n,
    # pick the prime p (not yet used) that minimizes 1/p among those that can cover n.
    # But "can cover n" means we can set r = n % p. Since r can be any value 0..p-1,
    # any prime can cover any n. So we just need to pick the cheapest prime not yet used.
    # That would be: use the largest primes first (smallest 1/p).
    #
    # But wait: if we use the largest primes first, each prime covers exactly 1 residue class,
    # which hits ~L/p integers in 1..L. With large p, that's few integers.
    #
    # Alternative: use small primes with carefully chosen residues to create overlaps
    # that cover consecutive integers.
    #
    # Let's try a different approach: 
    # 1. Start with an empty set of (p, r) pairs.
    # 2. For n = 1, 2, 3, ..., find the cheapest available prime p (min 1/p, i.e., max p)
    #    that hasn't been used, and assign r = n % p. Add (p, r) to the set.
    # 3. Continue until budget is exhausted or no primes left.
    # 4. Compute the covered length L.
    #
    # This is O(L * num_primes) which might be slow for L=100000.
    #
    # Better: precompute for each prime p the set of integers it covers given a residue r.
    # But r depends on which n we're covering.
    #
    # Let's try: sort primes in descending order (largest first). For each prime, 
    # assign a residue that covers the most currently-uncovered integers in 1..L_target.
    # But we don't know L_target ahead of time.
    #
    # Simpler greedy: 
    # - Maintain a boolean array covered[1..L_max]
    # - For n = 1 to L_max:
    #   - If covered[n], continue
    #   - Find the unused prime p that minimizes 1/p (i.e., largest p)
    #   - Set r = n % p, add (p, r) to solution
    #   - Mark all m in 1..L_max with m ≡ r (mod p) as covered
    #   - Subtract 1/p from budget
    #   - If budget < 0, stop
    # - Return the solution and compute L.
    #
    # This should be fast enough if we limit L_max to ~100000.
    
    L_max = 100000
    covered = bytearray(L_max + 1)  # covered[i] = 1 if i is covered
    
    # Sort primes in descending order (largest first, so 1/p is smallest)
    primes_desc = sorted(PRIMES_1000, reverse=True)
    used_primes = set()
    solution = []
    remaining_budget = Fraction(1, 1)
    
    # For efficiency, we'll iterate through n and for each uncovered n,
    # pick the largest unused prime.
    # To make this fast, we can precompute the next available prime.
    
    # Actually, let's just iterate n from 1 to L_max, and for each uncovered n,
    # find the largest unused prime. We can maintain a pointer.
    
    # Precompute: for each prime index, the prime value
    # We'll iterate primes_desc and keep track of which are used.
    
    # For speed, let's use a different approach:
    # For each n from 1 to L_max:
    #   if covered[n]: continue
    #   Find the largest p in primes_desc not in used_primes
    #   r = n % p
    #   Add (p, r) to solution
    #   Mark all m ≡ r (mod p) in 1..L_max as covered
    #   remaining_budget -= 1/p
    #   If remaining_budget < 0: break
    
    # To find the largest unused prime quickly, we can iterate primes_desc from the end
    # (smallest primes) and find the first unused one... wait, primes_desc is descending,
    # so primes_desc[0] is the largest. We want the largest unused, so we scan from index 0.
    
    # This could be slow if many primes are used. Let's use a different strategy:
    # Precompute for each n, the best prime to use. But that's complex.
    
    # Let's just do it naively and hope it's fast enough.
    # Number of primes up to 1000 is 168. So at most 168 primes can be used.
    # For each of the 168 primes, we mark ~L_max/p integers as covered.
    # Total work: sum over used primes of L_max/p, which is L_max * sum(1/p) <= L_max * 1 = 100000.
    # Plus the overhead of finding the largest unused prime each time.
    
    # Let's implement:
    primes_desc_sorted = sorted(PRIMES_1000, reverse=True)
    num_primes = len(primes_desc_sorted)
    is_used = [False] * num_primes
    
    for n in range(1, L_max + 1):
        if covered[n]:
            continue
        
        # Find the largest unused prime
        found = False
        for idx in range(num_primes):
            if not is_used[idx]:
                p = primes_desc_sorted[idx]
                is_used[idx] = True
                r = n % p
                solution.append([p, r])
                
                # Mark all m ≡ r (mod p) in 1..L_max as covered
                # m = r + k*p, for k such that 1 <= m <= L_max
                # If r == 0, m starts at p; else m starts at r
                start = r if r > 0 else p
                if start < 1:
                    start = 1
                for m in range(start, L_max + 1, p):
                    covered[m] = 1
                
                remaining_budget -= Fraction(1, p)
                found = True
                break
        
        if not found:
            break
        if remaining_budget < 0:
            # This last prime exceeded the budget; remove it
            solution.pop()
            is_used[idx] = False
            # Unmark the coverage? This is complex. Let's just break.
            break
    
    # Compute the actual covered length L
    L = 0
    for i in range(1, L_max + 1):
        if covered[i]:
            L = i
        else:
            break
    
    return solution

# Test
if __name__ == "__main__":
    sol = construct()
    print(f"Number of classes: {len(sol)}")
    # Compute covered length
    L_max = 100000
    covered = bytearray(L_max + 1)
    for p, r in sol:
        start = r if r > 0 else p
        for m in range(start, L_max + 1, p):
            covered[m] = 1
    L = 0
    for i in range(1, L_max + 1):
        if covered[i]:
            L = i
        else:
            break
    print(f"Covered length L: {L}")
    # Check budget
    total = sum(Fraction(1, p) for p, r in sol)
    print(f"Total budget used: {total}")
