import time
import math
import random
import heapq

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def construct():
    start_time = time.time()
    deadline = start_time + 290  # safety margin
    
    primes = _primes(N)
    
    # Build conflict graph: vertices are integers in [0, N]
    # Two vertices u, v conflict if adding both would complete a full residue class mod some prime p <= 5000
    # Specifically: for some prime p, the set {u, v} combined with existing points covers all residues mod p
    # But since we're building from scratch, we define conflict as:
    # u and v conflict if there exists a prime p such that u ≡ a (mod p), v ≡ b (mod p),
    # and {a, b} would be the last two residues needed... 
    # 
    # Actually, let me reconsider. The admissibility condition is:
    # For every prime p <= k (where k = |tuple|), some residue class mod p is missed.
    # 
    # A simpler conflict model: u and v conflict if for some prime p, 
    # u mod p and v mod p are such that including both forces all p residues to be present
    # in any set containing them plus other elements.
    #
    # Better approach: Use the greedy removal framework but with graph-based ordering.
    # The "conflict" between two points is that they share residue classes in a way that
    # makes it harder to satisfy admissibility.
    #
    # Let me use a different formulation: 
    # We want to find the largest subset S of [0,N] such that for all primes p <= |S|,
    # S does not cover all residues mod p.
    #
    # Greedy approach with graph coloring insight:
    # Process primes in order. For each prime p, if current set covers all residues,
    # we must remove one residue class. The key insight from graph coloring is to use
    # degree centrality: remove the residue class that has the most "connections" 
    # to other residue classes that we'll need later.
    #
    # Implementation: Use a priority queue based on a surrogate for future conflict.
    
    points = list(range(N + 1))
    point_set = set(points)
    
    # Precompute residues for all primes we might need
    # For efficiency, we'll compute on the fly but cache
    
    # Strategy: For each prime p (in increasing order), if all residues are present,
    # remove the residue class that minimizes future conflicts.
    # Future conflict proxy: number of larger primes q where this residue class 
    # "overlaps" with needed residues.
    
    # Simpler effective strategy: randomized greedy with multiple restarts
    # and smart tie-breaking based on density of remaining points
    
    best_result = None
    best_size = 0
    
    num_restarts = 50  # will adjust based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        random.seed(restart * 12345 + 7)
        
        current = list(range(N + 1))
        current_set = set(current)
        
        for p in primes:
            if time.time() > deadline - 5:
                break
                
            current_size = len(current)
            if p > current_size:
                break
            
            # Count residues
            counts = [0] * p
            for v in current:
                counts[v % p] += 1
            
            if 0 in counts:
                continue
            
            # All residues present - must remove one
            # Use degree-based heuristic: prefer removing residue classes
            # that have lower "degree" in the conflict graph
            # Approximation: prefer residues where removing them causes less
            # disruption to future primes
            
            # Compute a score for each residue class:
            # For residue r mod p, score = sum over larger primes q of 
            # (number of points in current that are ≡ r mod p AND ≡ s mod q for some s)
            # This is expensive, so use a simpler proxy:
            # score(r) = count[r] (original count) - penalty for being "central"
            
            # Simpler: weighted random selection with bias toward smaller counts
            # and randomness for exploration
            
            min_count = min(counts)
            max_count = max(counts)
            
            # Build weights: prefer smaller counts, add randomness
            weights = []
            for r in range(p):
                # Lower count = higher weight (more likely to be removed)
                # Add some randomness
                base = max_count - counts[r] + 1
                weight = base + random.random() * min_count
                weights.append(weight)
            
            # Normalize and sample
            total_weight = sum(weights)
            rand_val = random.random() * total_weight
            cumulative = 0
            target_residue = 0
            for r in range(p):
                cumulative += weights[r]
                if rand_val <= cumulative:
                    target_residue = r
                    break
            
            # Remove all points with this residue
            current = [v for v in current if v % p != target_residue]
        
        current_size = len(current)
        if current_size > best_size:
            best_size = current_size
            best_result = current
    
    if best_result is None:
        best_result = points
    
    return best_result
