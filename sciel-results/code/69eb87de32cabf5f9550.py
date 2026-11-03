<<<<<<< SEARCH
      # Greedy MIS
      best_set = []
      best_size = 0
      rng = random.Random(42)
      
      num_restarts = 0
      while time.time() < deadline - 10:
          perm = rng.permutation(N)
=======
      # Greedy MIS
      best_set = []
      best_size = 0
      rng = np.random.default_rng(42)
      
      num_restarts = 0
      while time.time() < deadline - 10:
          perm = rng.permutation(N)
>>>>>>> REPLACE
