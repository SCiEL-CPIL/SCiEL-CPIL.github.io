"""Local search over residue class assignments for admissible tuple maximization."""
import time
import numpy as np

N = 5000
DEADLINE = 295.0  # safety margin below 330s kill

def _primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.where(sieve)[0].tolist()

def construct():
    start_time = time.time()
    all_primes = _primes(N + 1)
    num_primes = len(all_primes)
    primes_arr = np.array(all_primes, dtype=np.int64)
    
    # x_vals: shape (N+1,)
    x_vals = np.arange(N + 1, dtype=np.int64)
    
    # Initialize residue choices randomly
    rng = np.random.default_rng(42)
    r = np.array([rng.integers(0, p) for p in all_primes], dtype=np.int64)
    
    # Compute initial ban_count: for each x, how many primes ban it
    # ban_count[x] = sum over i of (x_vals[x] % primes_arr[i] == r[i])
    # Vectorized: create (N+1, num_primes) boolean matrix
    # But this is (N+1)*num_primes which is ~3.35M, acceptable
    banned_matrix = (x_vals[:, None] % primes_arr[None, :]) == r[None, :]
    ban_count = banned_matrix.sum(axis=1)  # shape (N+1,)
    count = int((ban_count == 0).sum())
    
    best_count = count
    best_r = r.copy()
    
    # Local search: hill climbing with occasional simulated annealing
    temperature = 10.0
    cooling = 0.999
    no_improve = 0
    
    while time.time() - start_time < DEADLINE:
        # Pick a random prime index
        i = rng.integers(0, num_primes)
        p = all_primes[i]
        old_r = r[i]
        
        # Try all residue classes for this prime
        best_new_r = old_r
        best_new_count = count
        
        for new_r in range(p):
            if new_r == old_r:
                continue
            
            # Compute change in ban_count
            # was_banned: x_vals % p == old_r
            # is_banned: x_vals % p == new_r
            was_banned = (x_vals % p) == old_r
            is_banned = (x_vals % p) == new_r
            
            # Elements that lose ban from p: was_banned & ~is_banned
            lose_ban = was_banned & ~is_banned
            # Elements that gain ban from p: ~was_banned & is_banned
            gain_ban = ~was_banned & is_banned
            
            new_ban_count = ban_count.copy()
            new_ban_count[lose_ban] -= 1
            new_ban_count[gain_ban] += 1
            
            new_count = int((new_ban_count == 0).sum())
            
            if new_count > best_new_count:
                best_new_count = new_count
                best_new_r = new_r
        
        # Accept best move
        if best_new_r != old_r:
            # Update r, ban_count, count
            old_r = r[i]
            r[i] = best_new_r
            p = all_primes[i]
            was_banned = (x_vals % p) == old_r
            is_banned = (x_vals % p) == best_new_r
            lose_ban = was_banned & ~is_banned
            gain_ban = ~was_banned & is_banned
            ban_count[lose_ban] -= 1
            ban_count[gain_ban] += 1
            count = best_new_count
            
            if count > best_count:
                best_count = count
                best_r = r.copy()
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1
        
        # Simulated annealing: occasionally accept worse moves
        if no_improve > 100 and temperature > 0.1:
            # Pick a random new residue
            i = rng.integers(0, num_primes)
            p = all_primes[i]
            old_r = r[i]
            new_r = rng.integers(0, p)
            if new_r != old_r:
                was_banned = (x_vals % p) == old_r
                is_banned = (x_vals % p) == new_r
                lose_ban = was_banned & ~is_banned
                gain_ban = ~was_banned & is_banned
                
                new_ban_count = ban_count.copy()
                new_ban_count[lose_ban] -= 1
                new_ban_count[gain_ban] += 1
                new_count = int((new_ban_count == 0).sum())
                
                delta = new_count - count
                if delta >= 0 or rng.random() < np.exp(delta / temperature):
                    r[i] = new_r
                    ban_count = new_ban_count
                    count = new_count
                    if count > best_count:
                        best_count = count
                        best_r = r.copy()
                        no_improve = 0
                temperature *= cooling
        
        if no_improve > 500:
            break
    
    # Compute final tuple from best_r
    banned_matrix = (x_vals[:, None] % primes_arr[None, :]) == best_r[None, :]
    in_T = ~np.any(banned_matrix, axis=1)
    result = x_vals[in_T].tolist()
    
    return result
