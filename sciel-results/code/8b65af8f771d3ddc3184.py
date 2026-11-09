import time
import random
from collections import deque

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # Initial guess: Pawlik automaton (threshold N-1 = 11)
    a_map = [(i + 1) % N for i in range(N)]
    b_map = [0] * N
    for i in range(1, N):
        b_map[i] = i - 1
        
    best_a = a_map[:]
    best_b = b_map[:]
    best_score = -1

    # Precompute transition tables for the current best?
    # No, we need to recompute for each candidate.
    # But we can write a fast evaluator.

    def evaluate(a, b):
        """
        Computes the shortest reset word length for the automaton defined by a, b.
        Returns -1 if not synchronizing.
        """
        full_mask = (1 << N) - 1
        singleton_masks = {1 << i for i in range(N)}
        
        # Precompute next states for all masks
        # This is O(2^N * N) which is ~50k ops.
        next_a = [0] * (1 << N)
        next_b = [0] * (1 << N)
        
        for mask in range(1 << N):
            mask_a = 0
            mask_b = 0
            for i in range(N):
                if mask & (1 << i):
                    mask_a |= (1 << a[i])
                    mask_b |= (1 << b[i])
            next_a[mask] = mask_a
            next_b[mask] = mask_b
            
        # BFS from full_mask
        visited = {full_mask: 0}
        queue = deque([full_mask])
        
        while queue:
            current_mask = queue.popleft()
            dist = visited[current_mask]
            
            if current_mask in singleton_masks:
                return dist
                
            next_mask_a = next_a[current_mask]
            if next_mask_a not in visited:
                visited[next_mask_a] = dist + 1
                queue.append(next_mask_a)
                
            next_mask_b = next_b[current_mask]
            if next_mask_b not in visited:
                visited[next_mask_b] = dist + 1
                queue.append(next_mask_b)
                
        return -1 # Not synchronizing

    # Initial evaluation
    current_score = evaluate(best_a, best_b)
    best_score = current_score

    # Local Search
    while time.time() < deadline:
        # Create a copy of the current best
        cand_a = best_a[:]
        cand_b = best_b[:]
        
        # Perturb: change a random transition
        if random.random() < 0.5:
            idx = random.randint(0, N - 1)
            new_dest = random.randint(0, N - 1)
            if cand_a[idx] != new_dest:
                cand_a[idx] = new_dest
            else:
                continue # Skip if no change
        else:
            idx = random.randint(0, N - 1)
            new_dest = random.randint(0, N - 1)
            if cand_b[idx] != new_dest:
                cand_b[idx] = new_dest
            else:
                continue # Skip if no change
                
        # Evaluate candidate
        cand_score = evaluate(cand_a, cand_b)
        
        # Accept if better or equal (to allow plateaus)
        if cand_score > best_score:
            best_score = cand_score
            best_a = cand_a
            best_b = cand_b
            
        # Optional: If we reach the theoretical bound (121), we can stop
        if best_score == (N - 1) ** 2:
            break

    return [best_a, best_b]

if __name__ == "__main__":
    maps = construct()
    print("Constructed automaton.")
