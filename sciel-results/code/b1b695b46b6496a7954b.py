import numpy as np
import itertools
import random
import math
import time

def construct(q, n, radius):
    total_words = q ** n
    
    def to_int(word):
        val = 0
        for i in range(n):
            val = val * q + word[i]
        return val
        
    def to_word(val):
        word = [0] * n
        for i in range(n - 1, -1, -1):
            word[i] = val % q
            val //= q
        return word

    # Precompute balls
    balls = []
    for i in range(total_words):
        word = to_word(i)
        ball = [i]
        for pos in range(n):
            for delta in range(1, q):
                new_word = word[:]
                new_word[pos] = (word[pos] + delta) % q
                ball.append(to_int(new_word))
        balls.append(ball)
        
    # Initialize
    initial_size = 500
    codewords_list = random.sample(range(total_words), initial_size)
    codewords_set = set(codewords_list)
    
    cov = np.zeros(total_words, dtype=np.int32)
    for c in codewords_set:
        for w in balls[c]:
            cov[w] += 1
            
    uncovered_count = np.sum(cov == 0)
    
    M = 1000 
    T = 10.0
    T_min = 1e-6
    alpha = 0.995
    start_time = time.time()
    max_time = 110.0 # Leave 10s margin
    
    best_solution = codewords_set.copy()
    best_cost = len(codewords_set) + M * uncovered_count
    current_cost = best_cost
    
    all_codewords = list(range(total_words))
    
    while time.time() - start_time < max_time:
        if T < T_min:
            break
            
        move_type = random.randint(0, 2)
        delta_cost = 0
        
        if move_type == 0: # Remove
            if len(codewords_list) == 0:
                continue
            idx = random.randint(0, len(codewords_list) - 1)
            c = codewords_list[idx]
            
            # Calculate delta
            new_uncovered = 0
            for w in balls[c]:
                if cov[w] == 1:
                    new_uncovered += 1
            delta_uncovered = new_uncovered
            delta_cost = -1 + M * delta_uncovered
            
            if delta_cost < 0 or (random.random() < math.exp(-delta_cost / T)):
                # Apply move
                codewords_set.remove(c)
                # Remove from list efficiently
                codewords_list[idx] = codewords_list[-1]
                codewords_list.pop()
                
                for w in balls[c]:
                    cov[w] -= 1
                uncovered_count += delta_uncovered
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = codewords_set.copy()
                    
        elif move_type == 1: # Add
            c = random.choice(all_codewords)
            if c in codewords_set:
                continue
                
            new_uncovered = 0
            for w in balls[c]:
                if cov[w] == 0:
                    new_uncovered += 1
            delta_uncovered = -new_uncovered
            delta_cost = 1 + M * delta_uncovered
            
            if delta_cost < 0 or (random.random() < math.exp(-delta_cost / T)):
                # Apply move
                codewords_set.add(c)
                codewords_list.append(c)
                
                for w in balls[c]:
                    cov[w] += 1
                uncovered_count += delta_uncovered
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = codewords_set.copy()
                    
        else: # Swap
            if len(codewords_list) == 0:
                continue
            idx = random.randint(0, len(codewords_list) - 1)
            c_old = codewords_list[idx]
            c_new = random.choice(all_codewords)
            if c_new in codewords_set:
                continue
                
            # Calculate delta
            ball_old = balls[c_old]
            ball_new = balls[c_new]
            
            # To avoid set creation overhead, we can use a small array or just iterate
            # Since balls are small (17), sets are fine.
            set_old = set(ball_old)
            set_new = set(ball_new)
            
            only_old = set_old - set_new
            only_new = set_new - set_old
            
            delta_uncovered = 0
            for w in only_old:
                if cov[w] == 1:
                    delta_uncovered += 1
            for w in only_new:
                if cov[w] == 0:
                    delta_uncovered -= 1
                    
            delta_cost = M * delta_uncovered # Size unchanged
            
            if delta_cost < 0 or (random.random() < math.exp(-delta_cost / T)):
                # Apply move
                codewords_set.remove(c_old)
                codewords_set.add(c_new)
                
                # Update list
                codewords_list[idx] = c_new
                
                for w in ball_old:
                    cov[w] -= 1
                for w in ball_new:
                    cov[w] += 1
                    
                uncovered_count += delta_uncovered
                current_cost += delta_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = codewords_set.copy()
                    
        T *= alpha
        
    result = [to_word(c) for c in best_solution]
    return result
