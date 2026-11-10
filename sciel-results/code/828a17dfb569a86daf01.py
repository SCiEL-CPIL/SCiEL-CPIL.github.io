import time
import random
from itertools import combinations

def is_sidon_subset(subset):
    """Check if a subset is Sidon (all pair sums including diagonals are unique)."""
    sums = set()
    for i in range(len(subset)):
        for j in range(i, len(subset)):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def count_sidon_4_subsets(A):
    """Count the number of 4-element subsets of A that are Sidon."""
    count = 0
    for subset in combinations(A, 4):
        if is_sidon_subset(subset):
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 80s limit to be safe
    
    # Initial candidate: interval
    A = list(range(12))
    current_fitness = count_sidon_4_subsets(A)
    best_A = A[:]
    best_fitness = current_fitness
    
    # If we already have 0 Sidon 4-subsets, we are done (max Sidon size is 3)
    # But we should check if max Sidon size is even smaller? 
    # If no 4-subset is Sidon, max Sidon size is at most 3.
    # We should verify if there is a 3-subset Sidon. There always is (any 3 distinct ints 
    # with no 3-AP are Sidon? No, 0,1,3 is Sidon. 0,1,2 is not (0+2=1+1)).
    # So if best_fitness == 0, we return best_A.
    
    if best_fitness == 0:
        return best_A

    # Local Search
    # We will try to swap elements to reduce the number of Sidon 4-subsets.
    # To speed up, we can pre-calculate or use incremental updates, but 495 checks is fast.
    
    # Define search space: [0, 128]
    # We will randomly pick an index to change and a new value.
    
    # To be more systematic, we can try all swaps for a few iterations.
    # But random restarts might be better.
    
    # Let's try a simple hill climbing with random restarts.
    
    for restart in range(100): # Max 100 restarts
        if time.time() > deadline - 10:
            break
        
        # Randomize current A if not first restart
        if restart > 0:
            # Perturb the best solution found so far or start fresh?
            # Start fresh from interval but with some random swaps?
            # Or just random set?
            # Random set in [0, 128] is likely to have large Sidon subsets (sparse).
            # So we should stay close to dense sets.
            # Let's start from the interval and apply K random swaps.
            A = list(range(12))
            for _ in range(random.randint(0, 5)):
                idx = random.randint(0, 11)
                new_val = random.randint(0, 128)
                if new_val not in A:
                    A[idx] = new_val
            current_fitness = count_sidon_4_subsets(A)
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_A = A[:]
                if best_fitness == 0:
                    return best_A
        
        # Hill climbing
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try all possible swaps? 12 * 117 ~ 1400 swaps.
            # Each swap requires recounting 495 subsets. 1400 * 495 ~ 700k operations.
            # This is fast in Python if optimized? 
            # 700k * (check 4-subset) might be slow.
            # Let's limit the number of swaps to try per step.
            
            # Randomly select a few swaps to try
            for _ in range(20):
                idx = random.randint(0, 11)
                old_val = A[idx]
                new_val = random.randint(0, 128)
                if new_val in A:
                    continue
                A[idx] = new_val
                new_fitness = count_sidon_4_subsets(A)
                if new_fitness < current_fitness:
                    current_fitness = new_fitness
                    improved = True
                    if current_fitness < best_fitness:
                        best_fitness = current_fitness
                        best_A = A[:]
                        if best_fitness == 0:
                            return best_A
                else:
                    A[idx] = old_val
                    
        # If we improved, we might want to do a more thorough local search
        # But for now, this is a start.
        
    return best_A
