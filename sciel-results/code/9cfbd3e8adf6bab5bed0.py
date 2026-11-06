"""Combined approach: greedy with multiple orders + optimized local search for odd covering."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

def construct():
    base = 51975
    deadline = time.time() + 110  # safety margin before 120s limit (evaluator kills at 150s)
    
    # Get all odd divisors of 51975 greater than 1
    # 51975 = 3^3 * 5^2 * 7 * 11
    divisors = [d for d in range(3, base + 1, 2) if base % d == 0]
    
    def score_classes(classes):
        """Count uncovered residues."""
        covered = bytearray(base)
        for m, r in classes:
            covered[r::m] = b'\x01' * len(range(r, base, m))
        return covered.count(0)
    
    def greedy(order):
        """Greedy construction with given modulus order."""
        covered = bytearray(base)
        classes = []
        for m in order:
            # Find residue that covers most uncovered residues
            best_r = 0
            best_count = -1
            for r in range(m):
                count = covered[r::m].count(0)
                if count > best_count:
                    best_count = count
                    best_r = r
            classes.append([m, best_r])
            covered[best_r::m] = b'\x01' * len(range(best_r, base, m))
        return classes
    
    def local_search_optimized(classes):
        """Optimized local search using coverage depth to evaluate moves in O(1) per element."""
        best_classes = [list(c) for c in classes]
        n_classes = len(best_classes)
        
        # Initialize depth array
        depth = np.zeros(base, dtype=np.int32)
        for m, r in best_classes:
            depth[r::m] += 1
        
        current_uncovered = int(np.sum(depth == 0))
        
        improved = True
        passes = 0
        max_passes = 100
        while improved and passes < max_passes and time.time() < deadline:
            improved = False
            passes += 1
            for i in range(n_classes):
                m, r = best_classes[i]
                
                # Calculate potential gain for each new residue r'
                # We want to maximize (Gained - Lost)
                # Gained: residues in C_i(r') that are currently depth 0
                # Lost: residues in C_i(r) that are currently depth 1 (only covered by i)
                
                # Pre-compute Lost for current r
                # C_i(r) is r::m
                # Lost is count of depth[x] == 1 for x in r::m
                # We can compute this quickly
                lost_current = int(np.sum(depth[r::m] == 1))
                
                best_r_new = r
                best_delta = 0 # We only accept if delta > 0 (score decreases)
                
                for new_r in range(m):
                    if new_r == r:
                        continue
                    # Gained is count of depth[x] == 0 for x in new_r::m
                    gained = int(np.sum(depth[new_r::m] == 0))
                    # Note: If we move from r to new_r, the 'lost' part is fixed for the 'from' state.
                    # The 'gained' part depends on 'to' state.
                    # Delta = Gained - Lost
                    delta = gained - lost_current
                    
                    if delta > best_delta:
                        best_delta = delta
                        best_r_new = new_r
                
                if best_delta > 0:
                    # Perform the move
                    old_r = r
                    new_r = best_r_new
                    # Update depth
                    depth[old_r::m] -= 1
                    depth[new_r::m] += 1
                    # Update class
                    best_classes[i][1] = new_r
                    # Update score
                    current_uncovered -= best_delta
                    improved = True
                    if current_uncovered == 0:
                        return best_classes, current_uncovered
                        
        return best_classes, current_uncovered
    
    # Initial greedy with increasing order
    best_classes = greedy(divisors)
    best_score = score_classes(best_classes)
    
    # Try decreasing order
    classes_dec = greedy(divisors[::-1])
    score_dec = score_classes(classes_dec)
    if score_dec < best_score:
        best_score = score_dec
        best_classes = classes_dec
    
    # Try several random orders
    random.seed(42)
    num_random_tries = 0
    while time.time() < deadline and num_random_tries < 200:
        order = divisors[:]
        random.shuffle(order)
        classes = greedy(order)
        score = score_classes(classes)
        if score < best_score:
            best_score = score
            best_classes = classes
            if best_score == 0:
                break
        num_random_tries += 1
    
    # Local search on the best found
    if time.time() < deadline - 5:
        best_classes, best_score = local_search_optimized(best_classes)
    
    return best_classes
# EVOLVE-BLOCK-END
