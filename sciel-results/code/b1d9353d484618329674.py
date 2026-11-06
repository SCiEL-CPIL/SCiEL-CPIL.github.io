def construct():
      N = 40
      edges = []
      # Base cycle
      for i in range(N):
          edges.append((i, (i + 1) % N))
      
      # Chords from 0 to 2..20
      for k in range(2, 21):
          edges.append((0, k))
      
      # Ensure unique and sorted for consistency?
      # The task says "edges are simple and distinct".
      # My construction has distinct edges.
      
      # Cycles
      cycles = []
      
      # Length 3 to 21: 0-1-...-(L-1)-0
      # For length L, vertices are 0, 1, ..., L-1
      # The closing edge is (L-1, 0). This chord exists for L-1 in 2..20 => L in 3..21
      for L in range(3, 22):
          cycle = list(range(L))
          cycles.append(cycle)
          
      # Length 22 to 39: 0-k-...-39-0
      # Length L = 41 - k => k = 41 - L
      # Vertices: 0, k, k+1, ..., 39
      # Closing edge is (39, 0) which is in base cycle. Chord (0, k) exists for k in 2..20.
      # For L=22, k=19. For L=39, k=2.
      for L in range(22, 40):
          k = 41 - L
          cycle = [0] + list(range(k, 40))
          cycles.append(cycle)
          
      # Length 40: 0-1-...-39-0
      cycles.append(list(range(40)))
      
      return {'edges': edges, 'cycles': cycles}
