# M1: W[c*n+0]=1, W[(c+1)*n+1]=1 ✓
# M2: W[c*n+1]=1, W[(c+1)*n+1]=-1 ✗ (should be only W[c*n+1]=1)
# M3: W[(c+1)*n+0]=1, W[(c+1)*n+1]=1 ✗ (should be W[c*n+1]=1, W[(c+1)*n+1]=-1)
# M4: W[c*n+0]=1, W[c*n+1]=1 ✗ (should be W[c*n+0]=-1, W[(c+1)*n+0]=1)
# M5: W[c*n+0]=-1, W[(c+1)*n+0]=1 ✗ (should be W[c*n+0]=1)
# M6: W[(c+1)*n+1]=1 ✗ (should be W[(c+1)*n+0]=1, W[(c+1)*n+1]=1)
# M7: W[c*n+0]=1 ✗ (should be W[c*n+0]=1, W[(c+1)*n+1]=1)
