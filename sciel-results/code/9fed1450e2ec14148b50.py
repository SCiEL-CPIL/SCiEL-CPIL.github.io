   def generate_classic_order(n):
       if n == 1:
           return [(1,)]
       prev = generate_classic_order(n-1)
       curr = []
       for p in prev:
           # p is tuple of length n-1
           # Generate n permutations
           for k in range(n):
               # Q_k: insert n at position n-1-k?
               # Q_0: n at end. (p[0], ..., p[n-2], n)
               # Q_1: n at n-1. (p[1], ..., p[n-2], n, p[0])
               # ...
               # Q_{n-1}: n at start. (n, p[0], ..., p[n-2])
               
               # Let's trace indices.
               # Block: p[0] ... p[n-2] n p[0] ... p[n-2]
               # Window 0: p[0]...p[n-2]n
               # Window 1: p[1]...p[n-2]n p[0]
               # ...
               # Window n-1: n p[0]...p[n-2]
               
               # General formula for window i (0-indexed):
               # If i == 0: (p[0], ..., p[n-2], n)
               # If i > 0:
               #   The window starts at index i in the block p[0]...p[n-2]n...
               #   Wait, the block is P n P.
               #   Indices 0..n-2 are P. Index n-1 is n. Indices n..2n-3 are P.
               #   Window i starts at i.
               #   If i < n-1:
               #      Elements: p[i], ..., p[n-2], n, p[0], ..., p[i-1]
               #      Length: (n-1 - i) + 1 + i = n. Correct.
               #   If i == n-1:
               #      Elements: n, p[0], ..., p[n-2]
               #      Length: 1 + n-1 = n. Correct.
               
               if i == 0:
                   q = tuple(p) + (n,)
               elif i < n-1:
                   q = tuple(p[i:]) + (n,) + tuple(p[:i])
               else: # i == n-1
                   q = (n,) + tuple(p)
               
               curr.append(q)
       return curr
   