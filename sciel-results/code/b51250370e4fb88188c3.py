import numpy as np

def heuristics(prizes, weights):
         # Aggregate prizes to get a single score per item
         total_prize = prizes.sum(axis=0)
         
         # Calculate L2 norm of weights per item
         # This penalizes items with high weight in multiple constraints
         # more than L1 (sum) but less than L-inf (max)
         weight_norm = np.linalg.norm(weights, axis=0)
         
         # Return heuristic scores
         return total_prize / (weight_norm + 1e-9)
