import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Generate primes <= 1000
    def get_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    primes_arr = np.array(primes, dtype=np.int64)
    
    # Evaluation function: computes the largest L such that every integer 1..L 
    # is covered by at least 2 congruences.
    # We check up to a max limit, say 5000, which is well beyond 1000.
    MAX_L = 5000
    xs = np.arange(1, MAX_L + 1, dtype=np.int64)
    
    def evaluate(residues):
        # Vectorized evaluation
        diff = xs[None, :] - residues[:, None]
        mods = np.remainder(diff, primes_arr[:, None])
        hits = (mods == 0)
        coverage = np.sum(hits, axis=0)
        
        mask = coverage < 2
        if not np.any(mask):
            return MAX_L
        return int(np.argmax(mask))
    
    # Initial solution: Use the seed strategy
    seed_residues = np.array([((i + 2) // 2) % p for i, p in enumerate(primes)], dtype=np.int64)
    residues = seed_residues.copy()
    
    current_score = evaluate(residues)
    best_residues = residues.copy()
    best_score = current_score
    
    # Multi-residue Simulated Annealing
    # Instead of changing 1 residue, we change a batch of k residues at once.
    # This allows the search to move across plateaus where single moves yield no gain.
    
    T = 5.0  # Higher initial temperature for multi-moves
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Batch size strategy: start with larger batches, reduce as T drops
    # Or keep it fixed? Let's try a dynamic batch size.
    # Small batches (1-3) for fine tuning, large batches (5-10) for exploration.
    
    while time.time() < deadline:
        # Dynamic batch size: 
        # When T is high, take larger steps (more residues) to explore.
        # When T is low, take smaller steps to exploit.
        if T > 1.0:
            k = random.randint(3, 10)
        elif T > 0.1:
            k = random.randint(2, 5)
        else:
            k = random.randint(1, 3)
            
        # Select k random unique indices
        indices = np.random.choice(n_primes, size=k, replace=False)
        
        # Save old values
        old_residues = residues[indices].copy()
        
        # Generate new random residues for these indices
        new_residues = np.array([random.randint(0, primes[i] - 1) for i in indices], dtype=np.int64)
        
        # Apply changes
        residues[indices] = new_residues
        
        new_score = evaluate(residues)
        
        delta = new_score - current_score
        
        if delta > 0:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_residues = residues.copy()
        else:
            # Accept with probability exp(delta / T)
            if delta == 0:
                # For multi-moves, delta=0 is very common. 
                # We should be more aggressive in accepting to explore the plateau.
                # But if we accept too many, we might drift away from good areas.
                # Let's accept with a moderate probability, e.g., 0.5 or 1.0.
                # Actually, for multi-moves, delta=0 usually means we moved to an equivalent state.
                # Accepting helps escape local minima if the landscape is rugged.
                accept_prob = 0.8 
            else:
                # delta < 0
                # If T is small, this will be very small.
                try:
                    accept_prob = np.exp(delta / T)
                except OverflowError:
                    accept_prob = 0.0
                    
            if random.random() < accept_prob:
                current_score = new_score
            else:
                # Reject: revert
                residues[indices] = old_residues
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            # Optional: reheat if stuck?
            # Let's add a simple reheat mechanism: if no improvement in last N steps, boost T.
            # For simplicity, we'll just keep cooling.
            
        if time.time() > deadline - 5.0:
            break
            
    return best_residues.tolist()
