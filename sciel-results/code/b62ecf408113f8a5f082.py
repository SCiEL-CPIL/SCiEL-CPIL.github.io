# Inside the refinement loop, after the single-column check
# Check if column i is in the span of columns j and k
for i in range(r):
    if np.all(Ua[:, i] == 0) and np.all(Va[:, i] == 0) and np.all(Wa[:, i] == 0):
        continue
    for j in range(r):
        if j == i: continue
        for k in range(r):
            if k == i or k == j: continue
            # Check if Ua[:, i] is in span(Ua[:, j], Ua[:, k])
            # And same for V and W
            # This is expensive, so we only do it if r is small or time allows
            
            # We can check rank of 3 columns
            # If rank is 2, they are dependent.
            # We need to find the coefficients to zero out i.
            # Ua[:, i] = a * Ua[:, j] + b * Ua[:, k]
            # Solve for a, b using 2 non-zero rows.
            
            # This is complex. Let's stick to the single column check but with more alpha candidates.
            pass
