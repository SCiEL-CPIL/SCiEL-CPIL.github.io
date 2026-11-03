      def _voltage_lift_petersen(m, target_girth, deadline):
          # ... setup Petersen adj ...
          
          # 1. Generate all non-backtracking closed walks of length < target_girth
          walks = [] # List of lists of edge indices
          for start in range(10):
              # BFS/DFS to find closed walks
              # State: (node, prev, path_edges)
              # We only care about length < target_girth
              stack = [(start, -1, [])]
              while stack:
                  node, prev, path = stack.pop()
                  if len(path) >= target_girth:
                      continue
                  if len(path) > 0 and node == start:
                      # Found a closed walk
                      # Check if it's already in walks?
                      # We can just add it. Duplicates are okay for checking,
                      # but they slow down the "bad walk" check.
                      # To avoid duplicates, we can canonicalize the walk?
                      # Or just use a set of tuples of edges?
                      # For now, just add.
                      walks.append(path)
                      continue # Do not extend a closed walk?
                      # Actually, non-backtracking closed walks can be extended.
                      # But if we found a short one, we care about it.
                      # If we extend it, we get a longer one.
                      # We need ALL walks of length < g.
                      # So we should extend.
                      # But we must avoid infinite loops.
                      # Since we limit length, it's fine.
                      # However, if we extend, we might generate the same walk multiple times.
                      # E.g. A-B-C-A and A-B-C-A-B-C-A?
                      # The second is not "simple" in the base?
                      # Non-backtracking closed walk definition:
                      # A walk is non-backtracking if $v_{i-1} \neq v_{i+1}$.
                      # It does not have to be simple.
                      # So we must find all such walks.
                      
                  for neighbor, edge_idx in adj[node]:
                      if neighbor == prev:
                          continue
                      stack.append((neighbor, node, path + [edge_idx]))
          
          # Remove duplicates?
          # Convert to tuples for hashing
          unique_walks = set()
          for w in walks:
              unique_walks.add(tuple(w))
          walks = list(unique_walks)
          
          # 2. Map edge to walks
          edge_to_walks = {i: [] for i in range(15)}
          for i, w in enumerate(walks):
              for e in w:
                  edge_to_walks[e].append(i)
          
          # 3. Local Search
          weights = [random.randint(0, m - 1) for _ in range(15)]
          bad_walks = set(range(len(walks)))
          
          # Function to compute sum of a walk
          def walk_sum(w_idx):
              s = 0
              for e in walks[w_idx]:
                  s += weights[e]
              return s % m
          
          # Initialize bad_walks
          for i in range(len(walks)):
              if walk_sum(i) == 0:
                  bad_walks.add(i)
              else:
                  bad_walks.discard(i)
          
          while time.time() < deadline - 5:
              if not bad_walks:
                  # Success
                  n = 10 * m
                  edges = []
                  for i, (u, v) in enumerate(base_edges):
                      w = weights[i]
                      for j in range(m):
                          u_l = u * m + j
                          v_l = v * m + (j + w) % m
                          edges.append((u_l, v_l))
                  return n, edges
              
              # Pick a random bad walk
              w_idx = random.choice(list(bad_walks))
              # Pick a random edge in that walk
              e_idx = random.choice(walks[w_idx])
              # Change weight
              old_w = weights[e_idx]
              new_w = random.randint(0, m - 1)
              if new_w == old_w:
                  new_w = (new_w + 1) % m
              weights[e_idx] = new_w
              
              # Update bad_walks
              # Only walks containing e_idx are affected
              for w_i in edge_to_walks[e_idx]:
                  if walk_sum(w_i) == 0:
                      bad_walks.add(w_i)
                  else:
                      bad_walks.discard(w_i)
          
          return None
      