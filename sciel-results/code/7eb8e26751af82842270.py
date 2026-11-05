   def construct(n):
       if n <= 1:
           return "1"
       if n == 2:
           return "121"
       
       # The greedy algorithm generates a superpermutation.
       # We simulate the process.
       
       # State: current suffix of length n-1
       # We maintain a set of visited permutations (edges).
       
       all_syms = list(range(1, n+1))
       start_node = tuple(all_syms[:n-1])
       
       # Graph: node -> list of (next_node, edge_perm)
       # We precompute the next node for each possible next symbol.
       
       # To speed up, we note that from a node u, the next node is determined
       # by the symbol appended.
       
       # We use a stack for backtracking.
       # path = [start_node]
       # stack = [(start_node, iterator over neighbors)]
       
       # This is effectively a DFS.
       
       visited_edges = set()
       word = list(start_node)
       
       # We need to ensure we visit all edges.
       # The greedy choice: pick the neighbor that leads to the fewest unvisited edges?
       # Or the lexicographically smallest?
       
       # The "Greedy" superpermutation uses the rule:
       # Pick the smallest symbol not in the current suffix.
       # If that leads to a dead end (all edges from there are visited), backtrack.
       
       # Let's implement this rule.
       
       def get_next_nodes(node):
           # Returns list of (next_node, symbol)
           node_set = set(node)
           res = []
           for s in all_syms:
               if s not in node_set:
                   next_node = node[1:] + (s,)
                   res.append((next_node, s))
           return res

       # DFS with backtracking
       # We want to find a path that covers all edges.
       
       # Since n=7 is small, we can use a recursive DFS.
       # But depth is 5040, recursion limit might be an issue.
       
       # Iterative DFS:
       stack = [(start_node, iter(get_next_nodes(start_node)))]
       path = [start_node]
       
       # We need to record the word.
       # The word is the sequence of symbols.
       # word starts with start_node.
       
       # When we move from u to v via symbol s, we append s.
       
       # We stop when all edges are visited.
       
       # To ensure we find a valid path, we use the greedy heuristic:
       # Sort neighbors by the symbol (smallest first).
       
       # However, simple DFS might get stuck.
       # The "Greedy" algorithm is a specific DFS strategy.
       
       # Let's use the known result: The greedy algorithm produces a valid superpermutation.
       # We just need to implement it correctly.
       
       # I will use a simpler approach:
       # Generate the greedy sequence of permutations.
       
       perms = []
       current = tuple(range(1, n+1))
       visited = set()
       visited.add(current)
       perms.append(current)
       
       while len(perms) < n * factorial(n): # Upper bound
           # Find next
           # The next permutation is the one that overlaps maximally?
           # No, the greedy rule is specific.
           
           # For now, I will output the standard greedy string for n=7
           # by using a known library or hardcoded logic?
           # No, I must write code.
           
           # I will use the cycle decomposition but with a better ordering.
           # The cycles can be ordered by their "index" in the greedy traversal.
           
           # Actually, the current program is 0.82.
           # A simple greedy algorithm should be 1.0 or better.
           
           # I will replace the code with a standard greedy superpermutation generator.
           
           pass
   