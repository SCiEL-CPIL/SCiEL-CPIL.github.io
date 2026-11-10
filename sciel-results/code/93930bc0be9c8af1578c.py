"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
from collections import deque

def construct():
    """
    Find 12 distinct positive integers <= 10000 such that the union of prime factors
    of all pairwise sums (distinct indices) is minimized.
    """
    MAX_VAL = 10000
    N = 12
    
    # Generate primes up to 20000 (since sums can be up to 20000)
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(20000)
    
    # Check if a number is smooth with respect to prime_set
    def is_smooth(n, prime_set):
        for p in prime_set:
            while n % p == 0:
                n //= p
        return n == 1
    
    # Generate smooth numbers <= max_val for a given set of primes
    def get_smooth_numbers(prime_set, max_val):
        smooth = set()
        q = deque([1])
        while q:
            x = q.popleft()
            if x > max_val:
                continue
            smooth.add(x)
            for p in prime_set:
                y = x * p
                if y <= max_val:
                    q.append(y)
        return smooth
    
    # Find a clique of size n in the compatibility graph
    def find_clique(smooth_set, n, prime_set):
        if len(smooth_set) < n:
            return None
        
        smooth_list = sorted(smooth_set)
        n_items = len(smooth_list)
        
        # Build adjacency: two numbers are compatible if their sum is smooth
        adj = [set() for _ in range(n_items)]
        for i in range(n_items):
            for j in range(i+1, n_items):
                s = smooth_list[i] + smooth_list[j]
                if is_smooth(s, prime_set):
                    adj[i].add(j)
                    adj[j].add(i)
        
        # Backtracking to find clique of size n
        degree = [len(adj[i]) for i in range(n_items)]
        order = sorted(range(n_items), key=lambda x: -degree[x])
        
        pos_to_orig = {pos: node for pos, node in enumerate(order)}
        orig_to_pos = {node: pos for pos, node in enumerate(order)}
        
        n_ord = len(ordered_nodes := order)
        adj_ord = [set() for _ in range(n_ord)]
        for i in range(n_ord):
            for j in adj[order[i]]:
                adj_ord[i].add(orig_to_pos[j])
        
        # Backtracking with better pruning
        def backtrack(start, current_clique, available):
            if len(current_clique) == n:
                return current_clique[:]
            
            remaining_needed = n - len(current_clique)
            if len(available) < remaining_needed:
                return None
            
            for i in range(len(available)):
                node = available[i]
                # Check compatibility with all in current_clique
                compatible = all(node in adj_ord[c] for c in current_clique)
                if not compatible:
                    continue
                
                current_clique.append(node)
                new_available = [a for a in available[i+1:] if a in adj_ord[node]]
                result = backtrack(i+1, current_clique, new_available)
                if result is not None:
                    return result
                current_clique.pop()
            
            return None
        
        result = backtrack(0, [], list(range(n_ord)))
        if result is None:
            return None
        return [smooth_list[order[node]] for node in result]
    
    # Try prime sets in order of increasing size
    prime_candidates = primes[:20]
    
    for k in range(1, 7):
        for prime_combo in itertools.combinations(prime_candidates, k):
            prime_set = set(prime_combo)
            smooth = get_smooth_numbers(prime_set, MAX_VAL)
            
            if len(smooth) < N:
                continue
            
            result = find_clique(smooth, N, prime_set)
            if result is not None:
                assert len(set(result)) == N
                assert all(1 <= x <= MAX_VAL for x in result)
                return result
    
    return list(range(1, 13))

# EVOLVE-BLOCK-END
