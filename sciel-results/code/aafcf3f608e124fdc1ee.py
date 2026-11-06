      def construct():
          # ... initial setup ...
          best_network = initial_network[:]
          
          while time.time() < deadline:
              # ... pruning logic ...
              
              if len(network) < len(best_network) and len(network) > 0:
                  best_network = network[:]
              
              # Check if network is empty
              if len(network) == 0:
                  break
              
          return best_network
      