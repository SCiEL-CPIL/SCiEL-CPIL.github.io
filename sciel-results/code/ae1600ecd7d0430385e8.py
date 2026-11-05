import math
import time

def construct():
       start_time = time.time()
       deadline = start_time + 50.0 # 50s limit for safety
       
       max_val = 256
       numbers = list(range(2, max_val + 1))
       
       best_set = []
       best_score = 0.0
       
       def is_compatible(a, b):
           return math.lcm(a, b) > max_val
       
       def greedy_mws(candidates, weights):
           # candidates: list of ints
           # weights: dict int -> float
           # Returns set of ints
           selected = []
           # Sort by weight descending (value ascending)
           sorted_cands = sorted(candidates, key=lambda x: -weights[x])
           for c in sorted_cands:
               if all(is_compatible(c, s) for s in selected):
                   selected.append(c)
           return selected
       
       weights = {x: 1.0/x for x in numbers}
       
       # Strategy 1: Global Greedy
       s1 = greedy_mws(numbers, weights)
       score1 = sum(weights[x] for x in s1)
       if score1 > best_score:
           best_score = score1
           best_set = s1
           
       # Strategy 2: Anchor-based Greedy
       # Try each number as a forced anchor
       # We only need to try anchors that are likely to be in the optimal set.
       # Small numbers have high weight.
       # But large anchors allow including more numbers.
       # Let's try all p from 2 to 256.
       
       for p in numbers:
           if time.time() > deadline:
               break
           
           # Candidates compatible with p
           # p is compatible with b if lcm(p, b) > 256
           # If p | b, lcm = b <= 256 -> Conflict (unless b > 256, impossible)
           # So p cannot be with multiples of p.
           # If p !| b, lcm = p*b/gcd(p, b). Since gcd(p, b) < p (if p prime) or <= p/2?
           # Generally, if p !| b, lcm(p, b) >= b.
           # We need lcm(p, b) > 256.
           
           candidates = []
           for b in numbers:
               if b == p:
                   continue
               if math.lcm(p, b) > max_val:
                   candidates.append(b)
           
           # Run greedy on candidates
           # Note: p is not in candidates, we add it later
           s2 = greedy_mws(candidates, weights)
           s2.append(p)
           
           # Verify validity (greedy should ensure it, but check just in case)
           valid = True
           for i in range(len(s2)):
               for j in range(i+1, len(s2)):
                   if not is_compatible(s2[i], s2[j]):
                       valid = False
                       break
               if not valid: break
           
           if valid:
               score2 = sum(weights[x] for x in s2)
               if score2 > best_score:
                   best_score = score2
                   best_set = s2
                   
       # If best_set is empty, return [256] (valid, non-empty)
       if not best_set:
           best_set = [256]
           
       return best_set
