import sys
import time

def construct():
       # Generate primes up to 1000
       primes = [p for p in range(2, 1001) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
       primes.sort(reverse=True) # Descending order
       
       # Estimate max L.
       # Sum of ceil(L/p) >= 2L
       # For L=1000, sum approx 2200. 2L=2000. OK.
       # For L=2000, sum approx 4400. 2L=4000. OK.
       # Let's try L_max = 5000.
       L_max = 5000
       
       # Coverage array
       # cov[x] = count of primes covering x
       # We need cov[x] >= 2 for x in 1..L
       cov = [0] * (L_max + 1)
       
       residues = [0] * len(primes)
       
       # Greedy assignment
       for i, p in enumerate(primes):
           best_r = 0
           best_gain = -1
           # Try all residues? p can be up to 1000.
           # 168 primes * 1000 residues * (L/p) operations?
           # Total ops ~ sum(p * L/p) = sum(L) = 168 * 5000 = 840,000.
           # This is very fast.
           
           for r in range(p):
               gain = 0
               # Iterate x in 1..L_max such that x % p == r
               # x = r, r+p, r+2p ...
               # Note: if r==0, x starts at p. If r>0, x starts at r.
               start = r if r > 0 else p
               for x in range(start, L_max + 1, p):
                   if cov[x] < 2:
                       gain += 1
                   # Optimization: if gain + (remaining potential) < best_gain, break?
                   # Not easily done without precomputation.
                   
               if gain > best_gain:
                   best_gain = gain
                   best_r = r
                   
           residues[i] = best_r
           # Update coverage
           start = best_r if best_r > 0 else p
           for x in range(start, L_max + 1, p):
               cov[x] += 1
               
       # Find max L
       L = 0
       for x in range(1, L_max + 1):
           if cov[x] >= 2:
               L = x
           else:
               break
               
       # Map residues back to original prime order (increasing)
       # primes was sorted descending.
       # We need output for primes in increasing order.
       # Create a map: prime -> residue
       res_map = {}
       for p, r in zip(primes, residues):
           res_map[p] = r
           
       # Construct final list
       final_residues = []
       for p in sorted(res_map.keys()):
           final_residues.append(res_map[p])
           
       return final_residues
