def construct():
      import time
      from math import gcd
      
      start_time = time.time()
      deadline = start_time + 85.0
      max_mod = 128
      moduli = list(range(2, max_mod + 1))
      n = len(moduli)
      
      gcd_table = [[0]*n for _ in range(n)]
      for i in range(n):
          for j in range(n):
              gcd_table[i][j] = gcd(moduli[i], moduli[j])
              
      best_solution = []
      best_count = 0
      nodes = 0
      node_limit = 500000 # Increase limit if fast
      
      def check_residue(r, m_idx, assigned):
          for a_idx, r_a in assigned:
              g = gcd_table[m_idx][a_idx]
              if r % g == r_a % g:
                  return False
          return True
          
      def backtrack(assigned, remaining):
          nonlocal best_solution, best_count, nodes
          if time.time() > deadline or nodes > node_limit:
              return
          nodes += 1
          
          if len(assigned) > best_count:
              best_count = len(assigned)
              best_solution = [ [r, moduli[m_idx]] for m_idx, r in assigned ]
              
          if not remaining:
              return
              
          min_size = float('inf')
          min_idx = -1
          min_valids = []
          
          for m_idx in remaining:
              valids = []
              for r in range(moduli[m_idx]):
                  if check_residue(r, m_idx, assigned):
                      valids.append(r)
              if len(valids) < min_size:
                  min_size = len(valids)
                  min_idx = m_idx
                  min_valids = valids
                  if min_size == 0:
                      break
              
          if min_size == 0:
              remaining.remove(min_idx)
              backtrack(assigned, remaining)
              remaining.append(min_idx)
              return
              
          for r in min_valids:
              assigned.append((min_idx, r))
              remaining.remove(min_idx)
              backtrack(assigned, remaining)
              remaining.append(min_idx)
              assigned.pop()
              
          remaining.remove(min_idx)
          backtrack(assigned, remaining)
          remaining.append(min_idx)
          
      # Greedy init
      assigned = []
      remaining = list(range(n))
      for m_idx in remaining:
          valids = []
          for r in range(moduli[m_idx]):
              if check_residue(r, m_idx, assigned):
                  valids.append(r)
          if valids:
              assigned.append((m_idx, valids[0]))
              
      if len(assigned) > best_count:
          best_count = len(assigned)
          best_solution = [ [r, moduli[m_idx]] for m_idx, r in assigned ]
          
      # Backtrack from empty to find better?
      # Or backtrack from greedy?
      # Backtracking from empty with MRV is robust.
      # Let's just run backtrack from empty.
      
      backtrack([], list(range(n)))
      
      return best_solution
