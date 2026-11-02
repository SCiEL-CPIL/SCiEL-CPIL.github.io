import numpy as np
import time
import itertools

N = 50

def construct():
       start_time = time.time()
       deadline = start_time + 140.0
       
       def calc_score(h):
           h = np.maximum(h, 0)
           l1 = np.sum(h)
           if l1 < 1e-10: return 0.0
           l2_sq = np.sum(h**2)
           conv = np.convolve(h, h, mode='full')
           max_conv = np.max(conv)
           if max_conv < 1e-10: return 0.0
           return l2_sq / (l1 * max_conv)
       
       best_score = 0.0
       best_h = np.zeros(N)
       
       # Greedy construction of Sidon-like sets
       def greedy_build(seed_set, max_size=12):
           S = set(seed_set)
           # Precompute current sums?
           # Just recalculate for simplicity, N is small
           for _ in range(max_size):
               best_x = -1
               best_max_mult = float('inf')
               current_max_mult = get_max_mult(S)
               
               for x in range(N):
                   if x in S: continue
                   S_trial = S | {x}
                   m = get_max_mult(S_trial)
                   # We want to minimize max mult, but also increase K
                   # Score is K / M
                   # So we prefer K/M to be high
                   # But greedy step: just minimize M?
                   # If M stays 2, we can add more.
                   # If M becomes 3, K/M might drop.
                   # Let's evaluate K/M directly
                   score_trial = len(S_trial) / m if m > 0 else 0
                   # We want to maximize final score, so we pick the move that keeps potential high
                   # Simple heuristic: minimize M, then maximize K
                   if m < best_max_mult:
                       best_max_mult = m
                       best_x = x
                       best_score_trial = score_trial
                   elif m == best_max_mult:
                       # Tie break: prefer larger K? Or smaller x?
                       # If M is same, adding x increases K, so score increases.
                       # So any x that doesn't increase M is good.
                       # Pick smallest x to keep numbers small?
                       if best_x == -1 or x < best_x:
                           best_x = x
                           best_score_trial = score_trial
               
               if best_x != -1:
                   S.add(best_x)
               else:
                   break
           return S
       
       def get_max_mult(S):
           if not S: return 0
           sums = {}
           S_list = list(S)
           for i in S_list:
               for j in S_list:
                   s = i + j
                   sums[s] = sums.get(s, 0) + 1
           return max(sums.values())
       
       # Try various seeds
       seeds = [
           [0],
           [0, 1],
           [0, 1, 4],
           [0, 1, 4, 10],
           [0, 1, 4, 10, 18],
           # Known Sidon seeds
           [0, 1, 4, 10, 18, 23, 30],
           [1, 2, 5, 11, 21, 34], # Fibonacci-ish?
       ]
       
       for seed in seeds:
           if time.time() > deadline - 10: break
           S = greedy_build(seed)
           h = np.zeros(N)
           for i in S:
               h[i] = 1.0 / len(S)
           s = calc_score(h)
           if s > best_score:
               best_score = s
               best_h = h.copy()
               
       # Local search on the best set found
       # Represent best_h as set S
       S_best = set(np.nonzero(best_h)[0])
       
       improved = True
       while improved and time.time() < deadline - 5:
           improved = False
           current_score = calc_score(best_h)
           for i in list(S_best):
               for j in range(N):
                   if j in S_best: continue
                   # Swap i with j
                   S_trial = S_best.copy()
                   S_trial.remove(i)
                   S_trial.add(j)
                   h_trial = np.zeros(N)
                   for k in S_trial:
                       h_trial[k] = 1.0 / len(S_trial)
                   s_trial = calc_score(h_trial)
                   if s_trial > current_score + 1e-9:
                       S_best = S_trial
                       best_h = h_trial
                       current_score = s_trial
                       improved = True
                       
       return list(best_h)
