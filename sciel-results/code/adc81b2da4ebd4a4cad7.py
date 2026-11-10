"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def _sieve_primes(n):
    """Return list of primes <= n."""
    if n < 2:
        return []
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = 0
    is_prime[1] = 0
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = 0
    return [i for i in range(2, n + 1) if is_prime[i]]

def _is_coprime_to_all(a, others):
    """Check if a is coprime to all elements in others."""
    for b in others:
        if gcd(a, b) != 1:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 55.0  # safety margin before 90s kill

    n = 256

    # --- Candidate 1: primes + 1 (guaranteed valid) ---
    primes = _sieve_primes(n)
    prime_set = [1] + primes
    prime_sum = sum(prime_set)

    # --- Candidate 2: greedy descending (current best approach) ---
    selected = [1]
    for a in range(n, 1, -1):
        if time.time() - start_time > deadline:
            break
        if _is_coprime_to_all(a, selected):
            selected.append(a)
    greedy_sum = sum(selected)

    # Pick the better starting point
    best = selected if greedy_sum >= prime_sum else prime_set
    best_sum = max(greedy_sum, prime_sum)
    best_set = list(best)
    best_set = sorted(best_set)  # normalize

    # --- Local search: try to improve by adding or swapping ---
    # Build set for fast membership
    best_in_set = set(best_set)

    # Try to add any number not in best_set that is coprime to all in best_set
    # Process in descending order to prefer larger additions
    for a in range(n, 1, -1):
        if time.time() - start_time > deadline:
            break
        if a in best_in_set:
            continue
        if _is_coprime_to_all(a, best_set):
            best_set.append(a)
            best_in_set.add(a)
            best_sum += a

    # Now try more aggressive local search with proper state management
    
    max_swap_passes = 10
    for _pass in range(max_swap_passes):
        if time.time() - start_time > deadline:
            break
        improved = False
        
        # Try single-element swaps: replace b with larger a
        for b in sorted(best_set, reverse=True):
            if time.time() - start_time > deadline:
                break
            if b == 1:
                continue
            others = [x for x in best_set if x != b]
            for a in range(n, b, -1):
                if a in best_in_set:
                    continue
                if _is_coprime_to_all(a, others):
                    # Perform swap
                    best_set.remove(b)
                    best_in_set.discard(b)
                    best_set.append(a)
                    best_in_set.add(a)
                    best_sum += a - b
                    improved = True
                    break
        
        if not improved and time.time() - start_time < deadline:
            # Try remove-and-refill with proper tracking
            for b in sorted(best_set, reverse=True):
                if time.time() - start_time > deadline:
                    break
                if b == 1:
                    continue
                
                # Save current state
                saved_set = list(best_set)
                saved_in_set = set(best_in_set)
                saved_sum = best_sum
                
                # Remove b
                best_set.remove(b)
                best_in_set.discard(b)
                current_sum = best_sum - b
                
                # Try to add back numbers in descending order
                for a in range(n, 1, -1):
                    if time.time() - start_time > deadline:
                        break
                    if a in best_in_set:
                        continue
                    if _is_coprime_to_all(a, best_set):
                        best_set.append(a)
                        best_in_set.add(a)
                        current_sum += a
                
                # Check if this improved things
                if current_sum > best_sum:
                    best_sum = current_sum
                    improved = True
                else:
                    # Revert
                    best_set[:] = saved_set
                    best_in_set = set(saved_in_set)
                    best_sum = saved_sum
        
        if not improved:
            break

    # Final polish: try all possible single swaps multiple times
    if time.time() - start_time < deadline:
        for _ in range(10):
            if time.time() - start_time > deadline:
                break
            improved = False
            for b in sorted(best_set, reverse=True):
                if time.time() - start_time > deadline:
                    break
                if b == 1:
                    continue
                others = [x for x in best_set if x != b]
                for a in range(n, b, -1):
                    if a in best_in_set:
                        continue
                    if _is_coprime_to_all(a, others):
                        best_set.remove(b)
                        best_in_set.discard(b)
                        best_set.append(a)
                        best_in_set.add(a)
                        best_sum += a - b
                        improved = True
                        break
            if not improved:
                break

    # Ensure return type is list of Python ints
    result = [int(x) for x in best_set]
    return result
# EVOLVE-BLOCK-END
