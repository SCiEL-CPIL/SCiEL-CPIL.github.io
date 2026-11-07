<<<<<<< SEARCH
    # Candidate Heuristics
    cands = []
    
    # 1. Inverse Gap: High when pair fills bin tightly
    # gap = capacity - pair_sum
    # H = 1 / (gap + eps)
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = capacity - pair_sum
        h1 = np.where(fit, 1.0 / (gap + 1e-9), 0.0)
    cands.append(h1)
    
    # 2. Inverse Gap Squared
    with np.errstate(divide='ignore', invalid='ignore'):
        h2 = np.where(fit, 1.0 / ((gap + 1e-9) ** 2), 0.0)
    cands.append(h2)
    
    # 3. Pair Sum (Prefer larger pairs? Often bad, but let's see)
    h3 = np.where(fit, pair_sum, 0.0)
    cands.append(h3)
    
    # 4. Complement (Prefer smaller pairs? Leaves more room)
    h4 = np.where(fit, gap, 0.0)
    cands.append(h4)
    
    # 5. Product of Sizes
    h5 = np.where(fit, W[:, None] * W[None, :], 0.0)
    cands.append(h5)
    
    # 6. Ratio of Sizes (Balance?)
    with np.errstate(divide='ignore', invalid='ignore'):
        h6 = np.where(fit, np.minimum(W[:, None], W[None, :]) / (np.maximum(W[:, None], W[None, :]) + 1e-9), 0.0)
    cands.append(h6)
        
    # 7. Log Inverse Gap
    with np.errstate(divide='ignore', invalid='ignore'):
        h7 = np.where(fit, np.log1p(1.0 / (gap + 1e-9)), 0.0)
    cands.append(h7)
    
    # 8. Exponential Decay of Gap
    h8 = np.where(fit, np.exp(-gap / 10.0), 0.0)
    cands.append(h8)
=======
    # Candidate Heuristics
    cands = []
    
    # 1. Inverse Gap: High when pair fills bin tightly
    # gap = capacity - pair_sum
    # H = 1 / (gap + eps)
    with np.errstate(divide='ignore', invalid='ignore'):
        gap = capacity - pair_sum
        h1 = np.where(fit, 1.0 / (gap + 1e-9), 0.0)
    cands.append(h1)
    
    # 2. Inverse Gap Squared
    with np.errstate(divide='ignore', invalid='ignore'):
        h2 = np.where(fit, 1.0 / ((gap + 1e-9) ** 2), 0.0)
    cands.append(h2)
    
    # 3. Pair Sum (Prefer larger pairs? Often bad, but let's see)
    h3 = np.where(fit, pair_sum, 0.0)
    cands.append(h3)
    
    # 4. Complement (Prefer smaller pairs? Leaves more room)
    h4 = np.where(fit, gap, 0.0)
    cands.append(h4)
    
    # 5. Product of Sizes
    h5 = np.where(fit, W[:, None] * W[None, :], 0.0)
    cands.append(h5)
    
    # 6. Ratio of Sizes (Balance?)
    with np.errstate(divide='ignore', invalid='ignore'):
        h6 = np.where(fit, np.minimum(W[:, None], W[None, :]) / (np.maximum(W[:, None], W[None, :]) + 1e-9), 0.0)
    cands.append(h6)
        
    # 7. Log Inverse Gap
    with np.errstate(divide='ignore', invalid='ignore'):
        h7 = np.where(fit, np.log1p(1.0 / (gap + 1e-9)), 0.0)
    cands.append(h7)
    
    # 8. Exponential Decay of Gap
    h8 = np.where(fit, np.exp(-gap / 10.0), 0.0)
    cands.append(h8)
    
    # 9. Inverse Gap Cubed
    with np.errstate(divide='ignore', invalid='ignore'):
        h9 = np.where(fit, 1.0 / ((gap + 1e-9) ** 3), 0.0)
    cands.append(h9)
    
    # 10. Tight Fit Score: High when pair sum is close to capacity (e.g., > 0.8*capacity)
    tight_threshold = 0.8 * capacity
    h10 = np.where(fit & (pair_sum >= tight_threshold), 1.0 + (pair_sum - tight_threshold) / (capacity - tight_threshold), 0.0)
    cands.append(h10)
    
    # 11. Moderate Gap: Prefer gaps in a sweet spot (not too tight, not too loose)
    sweet_spot = 20.0  # Ideal gap around 20
    h11 = np.where(fit, 1.0 / (1.0 + np.abs(gap - sweet_spot)), 0.0)
    cands.append(h11)
    
    # 12. Size Complementarity: Prefer pairs where one is large and one is small
    w_max = W.max()
    w_min = W.min()
    size_range = w_max - w_min + 1e-9
    w_norm_i = (W[:, None] - w_min) / size_range
    w_norm_j = (W[None, :] - w_min) / size_range
    h12 = np.where(fit, np.abs(w_norm_i - w_norm_j), 0.0)
    cands.append(h12)
    
    # 13. Inverse Gap with Smoothing
    h13 = np.where(fit, 1.0 / (gap + 5.0), 0.0)
    cands.append(h13)
    
    # 14. Exponential with different scale
    h14 = np.where(fit, np.exp(-gap / 5.0), 0.0)
    cands.append(h14)
    
    # 15. Exponential with larger scale
    h15 = np.where(fit, np.exp(-gap / 20.0), 0.0)
    cands.append(h15)
    
    # 16. Sigmoid-like on gap
    h16 = np.where(fit, 1.0 / (1.0 + np.exp((gap - 15.0) / 5.0)), 0.0)
    cands.append(h16)
    
    # 17. Inverse Gap with floor
    h17 = np.where(fit, 1.0 / np.maximum(gap, 10.0), 0.0)
    cands.append(h17)
    
    # 18. Pair sum normalized
    h18 = np.where(fit, pair_sum / capacity, 0.0)
    cands.append(h18)
    
    # 19. Gap normalized
    h19 = np.where(fit, gap / capacity, 0.0)
    cands.append(h19)
    
    # 20. Combination: Inverse Gap * Size Balance
    h20 = np.where(fit, (1.0 / (gap + 1e-9)) * h6, 0.0)
    cands.append(h20)
    
    # 21. Combination: Tight Fit * Inverse Gap
    h21 = np.where(fit, h10 * (1.0 / (gap + 1e-9)), 0.0)
    cands.append(h21)
    
    # 22. Moderate Tightness: Reward gaps between 5 and 30
    h22 = np.where(fit & (gap >= 5) & (gap <= 30), 1.0 + (15.0 - np.abs(gap - 15.0)) / 15.0, 0.0)
    cands.append(h22)
    
    # 23. Inverse Gap with different epsilon
    h23 = np.where(fit, 1.0 / (gap + 0.1), 0.0)
    cands.append(h23)
    
    # 24. Quadratic inverse gap
    with np.errstate(divide='ignore', invalid='ignore'):
        h24 = np.where(fit, 1.0 / (gap ** 2 + 1e-9), 0.0)
    cands.append(h24)
    
    # 25. Sqrt inverse gap
    with np.errstate(divide='ignore', invalid='ignore'):
        h25 = np.where(fit, 1.0 / np.sqrt(gap + 1e-9), 0.0)
    cands.append(h25)
    
    # 26. Log gap (prefer larger gaps)
    with np.errstate(divide='ignore', invalid='ignore'):
        h26 = np.where(fit, np.log1p(gap), 0.0)
    cands.append(h26)
    
    # 27. Inverse log gap
    with np.errstate(divide='ignore', invalid='ignore'):
        h27 = np.where(fit, 1.0 / np.log1p(gap + 1e-9), 0.0)
    cands.append(h27)
    
    # 28. Pair sum squared
    h28 = np.where(fit, pair_sum ** 2, 0.0)
    cands.append(h28)
    
    # 29. Gap squared (prefer larger gaps)
    h29 = np.where(fit, gap ** 2, 0.0)
    cands.append(h29)
    
    # 30. Inverse gap with exponential smoothing
    h30 = np.where(fit, np.exp(-0.1 * gap) / (gap + 1e-9), 0.0)
    cands.append(h30)
    
    # 31. Weighted combination of inverse gap and pair sum
    h31 = np.where(fit, 0.5 * (1.0 / (gap + 1e-9)) + 0.5 * (pair_sum / capacity), 0.0)
    cands.append(h31)
    
    # 32. Weighted combination of tight fit and inverse gap
    h32 = np.where(fit, 0.3 * h10 + 0.7 * (1.0 / (gap + 1e-9)), 0.0)
    cands.append(h32)
    
    # 33. Prefer items that leave room for at least one more item
    min_item = W.min()
    h33 = np.where(fit & (gap >= min_item), 1.0 / (gap + 1e-9), 0.0)
    cands.append(h33)
    
    # 34. Prefer pairs that sum to at least 60% of capacity
    h34 = np.where(fit & (pair_sum >= 0.6 * capacity), 1.0 / (gap + 1e-9), 0.0)
    cands.append(h34)
    
    # 35. Prefer pairs that sum to at least 70% of capacity
    h35 = np.where(fit & (pair_sum >= 0.7 * capacity), 1.0 / (gap + 1e-9), 0.0)
    cands.append(h35)
    
    # 36. Prefer pairs that sum to at least 75% of capacity
    h36 = np.where(fit & (pair_sum >= 0.75 * capacity), 1.0 / (gap + 1e-9), 0.0)
    cands.append(h36)
    
    # 37. Prefer pairs that sum to at least 80% of capacity
    h37 = np.where(fit & (pair_sum >= 0.8 * capacity), 1.0 / (gap + 1e-9), 0.0)
    cands.append(h37)
    
    # 38. Prefer pairs that sum to at least 85% of capacity
    h38 = np.where(fit & (pair_sum >= 0.85 * capacity), 1.0 / (gap + 1e-9), 0.0)
    cands.append(h38)
    
    # 39. Prefer pairs that sum to at least 90% of capacity
    h39 = np.where(fit & (pair_sum >= 0.9 * capacity), 1.0 / (gap + 1e-9), 0.0)
    cands.append(h39)
    
    # 40. Balanced approach: moderate gap preference
    h40 = np.where(fit, 1.0 / (1.0 + np.abs(gap - 10.0) / 10.0), 0.0)
    cands.append(h40)
    
    # 41. Inverse gap with different smoothing
    h41 = np.where(fit, 1.0 / (gap + 2.0), 0.0)
    cands.append(h41)
    
    # 42. Inverse gap with different smoothing
    h42 = np.where(fit, 1.0 / (gap + 1.0), 0.0)
    cands.append(h42)
    
    # 43. Inverse gap with different smoothing
    h43 = np.where(fit, 1.0 / (gap + 0.5), 0.0)
    cands.append(h43)
    
    # 44. Inverse gap with different smoothing
    h44 = np.where(fit, 1.0 / (gap + 0.01), 0.0)
    cands.append(h44)
    
    # 45. Inverse gap with different smoothing
    h45 = np.where(fit, 1.0 / (gap + 0.001), 0.0)
    cands.append(h45)
    
    # 46. Exponential with scale 3
    h46 = np.where(fit, np.exp(-gap / 3.0), 0.0)
    cands.append(h46)
    
    # 47. Exponential with scale 7
    h47 = np.where(fit, np.exp(-gap / 7.0), 0.0)
    cands.append(h47)
    
    # 48. Exponential with scale 15
    h48 = np.where(fit, np.exp(-gap / 15.0), 0.0)
    cands.append(h48)
    
    # 49. Exponential with scale 25
    h49 = np.where(fit, np.exp(-gap / 25.0), 0.0)
    cands.append(h49)
    
    # 50. Exponential with scale 30
    h50 = np.where(fit, np.exp(-gap / 30.0), 0.0)
    cands.append(h50)
>>>>>>> REPLACE

