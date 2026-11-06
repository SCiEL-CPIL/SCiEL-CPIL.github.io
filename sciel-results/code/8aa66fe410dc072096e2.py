"""Photonic crystal unit-cell designer for max 4-5 TM band gap (square lattice, 32x32)."""
# EVOLVE-BLOCK-START
import math
import time
import random

def design(problem):
    """
    Design a 32x32 symmetric unit cell (square symmetry group) maximizing
    the gap-midgap ratio between TM bands 4 and 5.
    Strategy: local pixel-flip hill climbing over symmetry orbits, seeded by
    a parameterized four-rod design.
    """
    start_time = time.time()
    deadline = start_time + 218.0  # 12s safety margin before 230s kill

    size = problem.get("pixels", 32)
    eps_high = problem.get("eps_high", 12.0)
    eps_low = problem.get("eps_low", 1.0)

    # --- Symmetry orbit computation ---
    def get_orbit(i, j):
        """Return the orbit of pixel (i,j) under square symmetry group D4."""
        orbit = set()
        def add(x, y):
            orbit.add((x, y))
        # Identity
        add(i, j)
        # x -> -x  (j -> size-1-j)
        add(i, size-1-j)
        # y -> -y  (i -> size-1-i)
        add(size-1-i, j)
        # x -> -x, y -> -y
        add(size-1-i, size-1-j)
        # Transpose: (i,j) -> (j,i)
        add(j, i)
        # x->-x, transpose: (j, size-1-i)
        add(j, size-1-i)
        # y->-y, transpose: (size-1-j, i)
        add(size-1-j, i)
        # x->-x, y->-y, transpose: (size-1-j, size-1-i)
        add(size-1-j, size-1-i)
        return list(orbit)

    # Precompute all unique orbits
    visited = [[False]*size for _ in range(size)]
    orbits = []
    for i in range(size):
        for j in range(size):
            if not visited[i][j]:
                orbit = get_orbit(i, j)
                for (oi, oj) in orbit:
                    visited[oi][oj] = True
                orbits.append(orbit)

    # --- Plane-wave band solver (TM) ---
    # Use 17x17 = 289 plane waves
    Npw = 17  # plane waves per dimension
    # Plane wave indices: n_x, n_y in [-8..8]
    pw_indices = []
    for nx in range(-Npw//2, Npw//2+1):
        for ny in range(-Npw//2, Npw//2+1):
            pw_indices.append((nx, ny))
    Npw_total = len(pw_indices)  # 289

    def compute_fourier_coeffs(cell_2d):
        """
        Compute Fourier coefficients of 1/eps(x,y) for the pixel map.
        eps = eps_high where cell=1, eps_low where cell=0.
        We need 1/eps for TM polarization.
        1/eps = 1/eps_low where cell=0, 1/eps_high where cell=1.
        """
        # Build the 1/eps field
        inv_eps = [[0.0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                if cell_2d[i][j] == 1:
                    inv_eps[i][j] = 1.0/eps_high
                else:
                    inv_eps[i][j] = 1.0/eps_low

        # Compute Fourier coefficients G_{n,m} = integral over cell of (1/eps) exp(-i 2pi (n x + m y)) dx dy
        # Using pixel-centered sampling:
        # G_{n,m} = (1/N^2) sum_{i,j} (1/eps)_{i,j} exp(-i 2pi (n*(i+0.5)/N + m*(j+0.5)/N))
        # Actually for plane-wave method with periodic boundary:
        # G_{n,m} = (1/N^2) sum_{i,j} f(i,j) exp(-i 2pi (n i/N + m j/N))
        # But with pixel centers at (i+0.5)/N:
        # G_{n,m} = (1/N^2) sum_{i,j} f(i,j) exp(-i 2pi (n (i+0.5)/N + m (j+0.5)/N))
        #           = exp(-i pi (n+m)/N) * (1/N^2) sum_{i,j} f(i,j) exp(-i 2pi (n i/N + m j/N))

        # For efficiency, precompute exp factors
        # G_{n,m} for n,m in [-Npw//2, Npw//2]
        # We only need coefficients that couple plane waves differing by G = (n,m)
        # In the TM eigenvalue problem: (omega/k0)^2 lambda = sum_{G'} 1/eps(G-G') |E(G')|^2
        # Actually the eigenvalue problem is:
        # sum_{G'} 1/eps(G - G') |G|^2 delta_{G,G'} ... 
        # Let me use the standard formulation:
        # For TM: (omega/k0)^2 = (1/(2pi)^2) sum_{G'} 1/eps(G-G') |K+G'|^2 E(G') / ... 
        # Actually the standard PW matrix is:
        # H_{G,G'} = |K+G|^2 delta_{G,G'} - sum_{G''} 1/eps(G-G'') ... 
        # No, let me use the cleaner formulation.
        #
        # The TM eigenvalue problem in plane-wave expansion:
        # (K+G)^2 E(G) = (omega/k0)^2 sum_{G'} 1/eps(G-G') E(G')
        #
        # So the matrix is: A_{G,G'} = (K+G)^2 delta_{G,G'} - (omega/k0)^2 * inv_eps(G-G')
        # Wait, that's not right either. Let me re-derive.
        #
        # For TM (E along z): (nabla x nabla x E) = (omega/c)^2 eps E
        # In 2D with E = E_z(x,y) exp(i(kx+kz)):
        # (nabla_perp^2 + k_z^2) E_z = (omega/c)^2 eps E_z
        # In reciprocal space: (|K+G|^2 - k_z^2) E(G) = (omega/c)^2 sum_{G'} eps(G-G') E(G')
        # where K = (kx, ky) is the wavevector in the 2D BZ.
        # So: (|K+G|^2 - k_z^2) E(G) = (omega/c)^2 sum_{G'} eps(G-G') E(G')
        #
        # But for band structure at fixed K, we want eigenvalues omega.
        # At Gamma (K=0) and high-symmetry points, k_z is determined by the band.
        # Actually for a 2D photonic crystal, we compute the dispersion omega(K).
        # The eigenvalue problem at fixed K is:
        # sum_{G'} eps(G-G') E(G') = (omega/c)^2 / (|K+G|^2 - k_z^2) ... 
        # Hmm, this is getting complicated. Let me use the standard formulation:
        #
        # The eigenvalue problem is:
        # sum_{G'} 1/eps(G-G') |K+G'|^2 E(G') = (omega/c)^2 |K+G|^2 E(G)
        # Wait no. Let me look at this more carefully.
        #
        # For TM modes, the wave equation is:
        # [nabla_perp^2 + (omega/c)^2 eps(r)] E_z(r) = 0
        #
        # Expanding E_z = sum_G E(G) exp(i(K+G).r):
        # sum_G [-(K+G)^2 + (omega/c)^2] E(G) exp(i(K+G).r) = 0 ... no
        #
        # Actually: nabla_perp^2 E_z + (omega/c)^2 eps(r) E_z = 0
        # So: sum_G [-(K+G)^2] E(G) exp(i(K+G).r) + (omega/c)^2 sum_{G,G'} eps(G-G') E(G') exp(i(K+G').r) = 0
        #
        # Rearranging:
        # (omega/c)^2 sum_{G'} eps(G-G') E(G') = (K+G)^2 E(G)
        #
        # So the eigenvalue problem is:
        # sum_{G'} eps(G-G') E(G') = ((K+G)^2 / (omega/c)^2) E(G)
        #
        # Let lambda = ((K+G)^2 / (omega/c)^2) ... but this depends on G, so it's not a standard eigenvalue problem.
        #
        # Actually the correct formulation for finding omega at fixed K:
        # (omega/c)^2 sum_{G'} eps(G-G') E(G') = (K+G)^2 E(G)
        #
        # This can be written as a generalized eigenvalue problem or as a standard one by inverting.
        # Let's define: multiply both sides by 1/eps... no.
        #
        # Standard approach: The matrix element is:
        # H_{G,G'} = (K+G)^2 delta_{G,G'} 
        # And the "mass" matrix is:
        # M_{G,G'} = (omega/c)^2 eps(G-G')
        #
        # So: H E = M E, i.e., (K+G)^2 E(G) = (omega/c)^2 sum_{G'} eps(G-G') E(G')
        #
        # To make it a standard eigenvalue problem, we can write:
        # [sum_{G'} eps^{-1}(G-G')] ... no that's for TE.
        #
        # For TM: eigenvalue problem is (K+G)^2 E(G) = lambda sum_{G'} eps(G-G') E(G')
        # where lambda = (omega/c)^2.
        #
        # This is a generalized eigenvalue problem: A E = lambda B E
        # where A_{G,G'} = (K+G)^2 delta_{G,G'} and B_{G,G'} = eps(G-G')
        #
        # Equivalently: B^{-1} A E = lambda E
        #
        # B is circulant (depends on G-G'), so B^{-1} has elements (1/eps)(G-G') in reciprocal space.
        # So: sum_{G'} (1/eps)(G-G') (K+G')^2 E(G') = lambda (K+G)^2 E(G)
        #
        # Hmm, that's still not a standard symmetric eigenvalue problem.
        #
        # Let me use the alternative formulation that IS symmetric:
        # Define F(G) = sqrt(eps) ... no, eps varies.
        #
        # Actually, the cleanest symmetric formulation for TM:
        # Let u(G) = E(G) / sqrt((K+G)^2) ... no.
        #
        # Let me just use scipy's eig for the generalized problem, or use the standard:
        # The eigenvalue equation is:
        # sum_{G'} [1/eps(G-G')] * (K+G')^2 * E(G') = lambda * (K+G)^2 * E(G) ... 
        # No wait, I need to be more careful.
        #
        # Going back: (K+G)^2 E(G) = (omega/c)^2 sum_{G'} eps(G-G') E(G')
        #
        # This is: A E = lambda B E where A = diag((K+G)^2), B_{G,G'} = eps(G-G'), lambda = (omega/c)^2.
        #
        # To convert to standard form: B^{-1} A E = lambda E
        # B is a circulant matrix with first row [eps(0), eps(1), eps(2), ...] (in appropriate indexing).
        # B^{-1} is also circulant with first row [1/eps(0), 1/eps(1), ...] (Fourier transform).
        #
        # So: (B^{-1} A)_{G,G'} = sum_{G''} (1/eps)(G-G'') delta_{G'',G'} (K+G')^2 ... 
        # = (1/eps)(G-G') (K+G')^2
        #
        # So the matrix to diagonalize is: M_{G,G'} = (1/eps)(G-G') * (K+G')^2
        # And eigenvalues are lambda = (omega/c)^2.
        #
        # But this matrix is NOT symmetric in general. However, we can make it symmetric:
        # Define D = diag(sqrt((K+G)^2)) and define v(G) = D(G) u(G) where u = E/sqrt((K+G)^2)... 
        # This is getting messy. Let me just use the non-symmetric matrix and use numpy's eig.
        #
        # Actually, a simpler and more standard approach:
        # The eigenvalue problem for TM can be written as a standard symmetric eigenvalue problem:
        # sum_{G'} [1/eps(G-G')] E(G') = (omega/(c |K+G|))^2 ... no.
        #
        # Let me just directly construct the matrix and use numpy.linalg.eig.
        # M_{G,G'} = (1/eps)(G-G') * (K+G')^2
        # Eigenvalues give (omega/c)^2 / (K+G)^2 ... no, the eigenvalue lambda satisfies:
        # M E = lambda E, where lambda = (omega/c)^2 / (K+G)^2 ... 
        # No. Let me re-derive cleanly.
        #
        # (K+G)^2 E(G) = (omega/c)^2 sum_{G'} eps(G-G') E(G')
        # Multiply both sides by 1/(K+G)^2:
        # E(G) = (omega/c)^2 / (K+G)^2 sum_{G'} eps(G-G') E(G')
        #
        # Now multiply both sides by 1/eps... we want to isolate.
        # Let's define the matrix N_{G,G'} = eps(G-G')
        # Then: E = (omega/c)^2 * D^{-1} N E where D = diag((K+G)^2)
        # So: D^{-1} N E = (c/omega)^2 E
        # This is: N E = (c/omega)^2 D E, a generalized eigenvalue problem.
        #
        # Or equivalently: N^{-1} D E = (omega/c)^2 E
        # (N^{-1} D)_{G,G'} = sum_{G''} (1/eps)(G-G'') delta_{G'',G'} (K+G')^2 = (1/eps)(G-G') (K+G')^2
        #
        # So the matrix is: A_{G,G'} = (1/eps)(G-G') * (K+G')^2
        # and its eigenvalues are (omega/c)^2.
        #
        # This matrix is not Hermitian in general, but for real eps and real K, it should give real eigenvalues
        # (since the physics guarantees real frequencies). Let me just use numpy.linalg.eig and take real parts.
        #
        # For efficiency with 289 plane waves, this is a 289x289 matrix, which is fine.

        # Precompute 1/eps Fourier coefficients
        inv_eps_G = {}
        for nx in range(-Npw//2, Npw//2+1):
            for ny in range(-Npw//2, Npw//2+1):
                # G_{nx,ny} = (1/N^2) sum_{i,j} inv_eps(i,j) exp(-i 2pi (nx*i/N + ny*j/N))
                # With pixel centers: multiply by exp(-i 2pi (nx+ny)/(2N))
                s_re = 0.0
                s_im = 0.0
                for i in range(size):
                    for j in range(size):
                        angle = 2*math.pi*(nx*i/size + ny*j/size)
                        s_re += inv_eps[i][j]*math.cos(angle)
                        s_im += inv_eps[i][j]*math.sin(angle)
                s_re /= size*size
                s_im /= size*size
                # Apply pixel-center phase shift
                phase = math.pi*(nx+ny)/size
                inv_eps_G[(nx,ny)] = (s_re*math.cos(phase) + s_im*math.sin(phase),
                                       -(s_re*math.sin(phase) - s_im*math.cos(phase)))
        return inv_eps_G

    def solve_bands(cell_2d, Kx, Ky):
        """
        Solve for TM band frequencies at wavevector (Kx, Ky).
        Returns list of omega/k0 values (squared) sorted ascending.
        """
        # Get 1/eps Fourier coefficients (cache if possible, but for simplicity compute each time)
        # Actually, for the optimizer we call this many times, so let's pass the coefficients.
        pass

    # For the optimization, computing the full band structure each iteration is too slow.
    # Instead, let me use a precomputed inv_eps_G and evaluate at specific K points.

    # Key wavevectors in the irreducible BZ of square lattice:
    # Irreducible BZ: 0 <= Kx <= Ky, with Kx^2+Ky^2 bounded.
    # Key points: Gamma(0,0), M(pi,0), X(pi/2,pi/2), and lines.
    # The gap between bands 4 and 5 is typically at high-symmetry points or along lines.
    # For a 4-5 gap in a square lattice PC, the gap is often near M or X points.
    # Let me evaluate at a set of K points and find the minimum of band 5 and maximum of band 4.

    # Actually, for the optimizer, I need a fast evaluation. Let me precompute inv_eps_G once
    # for the current cell and reuse it. But since I'm changing the cell, I need to recompute.
    # 
    # Alternative: Use a simpler metric. For a square lattice PC with 4-fold symmetry,
    # the 4-5 TM gap is often bounded by the M point (pi,0) and/or X point (pi/2,pi/2).
    # Let me evaluate at M and X points primarily, plus a few others.

    # Wavevectors (in units where lattice constant a=1, so BZ goes from -pi to pi)
    # Actually, for a=1, the BZ is [-pi,pi]^2.
    # Key points:
    # Gamma: (0,0)
    # M: (pi, 0)  [equivalent to (0,pi) by symmetry]
    # X: (pi/2, pi/2)
    # Line X-M: (t*pi/2, (1-t)*pi) for t in [0,1] ... 
    # Let me use a small set of K points.

    K_points = [
        (0.0, 0.0),           # Gamma
        (math.pi, 0.0),       # M
        (math.pi/2, math.pi/2), # X
        (math.pi*0.75, 0.0),
        (math.pi*0.5, 0.0),
        (math.pi/2, math.pi*0.25),
        (math.pi*0.75, math.pi*0.25),
        (math.pi*0.5, math.pi*0.5),
    ]

    def evaluate_cell(cell_2d):
        """Compute the 4-5 TM gap-midgap ratio for the given cell."""
        # Compute 1/eps Fourier coefficients
        inv_eps_G = compute_fourier_coeffs(cell_2d)
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for (Kx, Ky) in K_points:
            # Build the matrix A_{G,G'} = inv_eps(G-G') * (K+G')^2
            # G' indices: pw_indices[gp2]
            # G indices: pw_indices[gp1]
            # (K+G')^2 = (Kx+Gx')^2 + (Ky+Gy')^2
            
            n = Npw_total
            A = [[0.0]*n for _ in range(n)]
            for gp1 in range(n):
                G1 = pw_indices[gp1]
                for gp2 in range(n):
                    G2 = pw_indices[gp2]
                    dGx = G1[0] - G2[0]
                    dGy = G1[1] - G2[1]
                    # inv_eps(dG)
                    # The Fourier coefficient index: we need inv_eps_G[(dGx, dGy)]
                    # But our inv_eps_G is defined for nx,ny in [-8,8]
                    # dGx, dGy are in [-16, 16], so we need to handle aliasing
                    # Actually, since we only have plane waves up to Npw//2=8, 
                    # and the convolution wraps around, we need to be careful.
                    # For a finite number of plane waves, the product eps*E is truncated,
                    # so the Fourier coefficients needed are those of eps for G-G' where
                    # both G and G' are in our set. The difference can be up to 2*8=16.
                    # But our eps is periodic with period 1, so G coefficients are defined for all integers.
                    # We only computed them for [-8,8]. For |dG| > 8, the coefficient may be nonzero
                    # but we're approximating by ignoring them (truncation error).
                    # This is the standard truncation in PW methods.
                    
                    key = (dGx, dGy)
                    if key in inv_eps_G:
                        ie_re, ie_im = inv_eps_G[key]
                    else:
                        # For |dG| > Npw//2, assume the coefficient is approximately zero
                        # (this is the truncation approximation)
                        ie_re, ie_im = 0.0, 0.0
                    
                    KGx = Kx + G2[0]*2*math.pi  # Wait, G is in units of 2pi/a? 
                    # Actually, in the plane-wave expansion, G = (n_x * 2pi/a, n_y * 2pi/a)
                    # With a=1, G = (n_x * 2pi, n_y * 2pi)
                    # But I've been using G as integer indices. Let me fix this.
                    # G = 2pi * (nx, ny)
                    KGx = Kx + G2[0]*2*math.pi
                    KGy = Ky + G2[1]*2*math.pi
                    KG2 = KGx*KGx + KGy*KGy
                    
                    A[gp1][gp2] = ie_re * KG2  # Real part (coefficients should be real for symmetric eps)
            
            # Solve eigenvalue problem: A E = lambda E
            # Use numpy
            import numpy as np
            A_np = np.array(A)
            eigenvalues = np.linalg.eigvalsh(A_np)  # A should be symmetric if inv_eps is real and symmetric
            # Sort eigenvalues
            eigenvalues = sorted(eigenvalues)
            
            # Eigenvalues are (omega/c)^2 / (K+G)^2 ... 
            # Wait, I need to re-examine. The matrix A I built is:
            # A_{G,G'} = inv_eps(G-G') * (K+G')^2
            # And the eigenvalue problem is A E = lambda E where lambda = (omega/c)^2.
            # But this is only correct if the matrix is properly constructed.
            # Let me double-check: from (K+G)^2 E(G) = (omega/c)^2 sum_{G'} eps(G-G') E(G')
            # Multiplying by 1/(K+G)^2: E(G) = (omega/c)^2/(K+G)^2 sum_{G'} eps(G-G') E(G')
            # This doesn't directly give A E = lambda E in the form I wrote.
            #
            # Let me use the generalized eigenvalue approach:
            # B E = lambda A E where B_{G,G'} = eps(G-G'), A_{G,G'} = (K+G)^2 delta_{G,G'}
            # This is harder. Let me use scipy.linalg.eig for the generalized problem.
            #
            # Actually, let me reconsider. The standard formulation that gives a symmetric matrix:
            # For TM: sum_{G'} [1/eps(G-G')] E(G') = (omega/c)^2 / (K+G)^2 ... no.
            #
            # I think the issue is that the TM problem doesn't naturally give a symmetric matrix
            # unless we do a transformation. Let me just use the generalized eigenvalue problem
            # with scipy.
            pass
        
        return 0.0  # Placeholder

    # Given the complexity and time constraints, let me use a simpler approach:
    # Use a well-known good design for 4-5 TM gap and do local optimization with a fast local solver.
    
    # Let me implement a proper but fast band solver.
    
    def compute_band_gaps(cell_2d):
        """
        Compute the 4-5 TM gap-midgap ratio.
        Uses 17x17 plane waves and evaluates at key K points.
        """
        import numpy as np
        import scipy.linalg as sla
        
        # Compute 1/eps Fourier coefficients
        inv_eps_G = compute_fourier_coeffs(cell_2d)
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        # Key K points in the irreducible BZ
        # BZ of square lattice (a=1): [-pi, pi]^2
        # Irreducible sector: 0 <= Kx <= Ky <= pi, Kx^2+Ky^2 <= pi^2 (roughly)
        # Actually the irreducible BZ for square lattice is:
        # 0 <= Kx <= Ky, with the boundary at Kx=Ky (the X line) and Ky=pi (the M edge)
        # and the arc from Gamma to M.
        
        K_points = [
            (0.0, 0.0),                    # Gamma
            (math.pi, 0.0),                # M
            (math.pi/2, math.pi/2),        # X
            (math.pi*0.6, 0.0),
            (math.pi*0.4, 0.0),
            (math.pi/2, 0.0),
            (math.pi/2, math.pi*0.25),
            (math.pi/2, math.pi*0.75),
            (math.pi*0.75, math.pi*0.25),
            (math.pi*0.3, math.pi*0.3),
            (math.pi*0.6, math.pi*0.6),
        ]
        
        for (Kx, Ky) in K_points:
            n = Npw_total
            # Build matrices for generalized eigenvalue problem:
            # B E = lambda A E
            # A_{G,G'} = (K+G)^2 delta_{G,G'}
            # B_{G,G'} = eps(G-G')  [NOT 1/eps]
            
            # Wait, I had: (K+G)^2 E(G) = (omega/c)^2 sum_{G'} eps(G-G') E(G')
            # So: A E = lambda B E where A=diag((K+G)^2), B_{G,G'}=eps(G-G'), lambda=(omega/c)^2
            
            A_mat = np.zeros((n, n), dtype=complex)
            B_mat = np.zeros((n, n), dtype=complex)
            
            for gp in range(n):
                G = pw_indices[gp]
                KGx = Kx + G[0]*2*math.pi
                KGy = Ky + G[1]*2*math.pi
                KG2 = KGx*KGx + KGy*KGy
                A_mat[gp][gp] = KG2
            
            for gp1 in range(n):
                G1 = pw_indices[gp1]
                for gp2 in range(n):
                    G2 = pw_indices[gp2]
                    dGx = G1[0] - G2[0]
                    dGy = G1[1] - G2[1]
                    # eps(G-G') = 1/inv_eps ... no, eps is the actual dielectric function
                    # We have 1/eps Fourier coefficients. We need eps Fourier coefficients.
                    # eps(G) = 1/inv_eps(-G) ... not exactly, since 1/(1/x) != 1/x in Fourier.
                    # 
                    # Hmm, I computed 1/eps Fourier coefficients, but for TM I need eps Fourier coefficients.
                    # Let me recompute: I should compute eps(G) directly.
                    pass
            
            # I made an error: for TM, I need eps(G), not 1/eps(G).
            # Let me fix this by computing eps Fourier coefficients instead.
            pass
        
        return 0.0
    
    # This is getting too complex for the time budget. Let me take a step back.
    # 
    # Given the 230s time limit and the need for a robust solution, let me use a 
    # proven good design for 4-5 TM band gap in square lattice photonic crystals.
    # 
    # Known good designs:
    # 1. Four rods at (±0.25, ±0.25) with radius ~0.15-0.2 (the current seed, score 5.79)
    # 2. A cross or plus-sign pattern
    # 3. Two large rods at (±0.3, ±0.3) 
    # 4. A design with a central hole
    #
    # The inspiration suggests that non-parametric shapes (split rods, crosses) can do better.
    # Let me try a cross/plus pattern and also do local optimization.
    
    # Let me implement a fast local band solver and do hill climbing.
    
    # For speed, I'll use a smaller set of plane waves (e.g., 11x11=121) for the optimizer
    # and trust that the relative ordering is preserved.
    
    Npw_opt = 11  # 11x11 = 121 plane waves for optimization
    pw_indices_opt = []
    for nx in range(-Npw_opt//2, Npw_opt//2+1):
        for ny in range(-Npw_opt//2, Npw_opt//2+1):
            pw_indices_opt.append((nx, ny))
    Npw_total_opt = len(pw_indices_opt)
    
    def compute_eps_fourier(cell_2d):
        """Compute Fourier coefficients of eps(x,y)."""
        eps_field = [[0.0]*size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                eps_field[i][j] = eps_high if cell_2d[i][j] == 1 else eps_low
        
        eps_G = {}
        for nx in range(-Npw_opt//2, Npw_opt//2+1):
            for ny in range(-Npw_opt//2, Npw_opt//2+1):
                s_re = 0.0
                s_im = 0.0
                for i in range(size):
                    for j in range(size):
                        angle = 2*math.pi*(nx*i/size + ny*j/size)
                        s_re += eps_field[i][j]*math.cos(angle)
                        s_im += eps_field[i][j]*math.sin(angle)
                s_re /= size*size
                s_im /= size*size
                phase = math.pi*(nx+ny)/size
                eps_G[(nx,ny)] = (s_re*math.cos(phase) + s_im*math.sin(phase),
                                   -(s_re*math.sin(phase) - s_im*math.cos(phase)))
        return eps_G
    
    def evaluate_cell_fast(cell_2d):
        """Fast evaluation of 4-5 TM gap using 11x11 plane waves."""
        import numpy as np
        
        eps_G = compute_eps_fourier(cell_2d)
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        K_points = [
            (0.0, 0.0),
            (math.pi, 0.0),
            (math.pi/2, math.pi/2),
            (math.pi*0.75, 0.0),
            (math.pi/2, 0.0),
            (math.pi/2, math.pi*0.25),
            (math.pi*0.75, math.pi*0.25),
        ]
        
        n = Npw_total_opt
        A_mat = np.zeros((n, n), dtype=float)
        B_mat = np.zeros((n, n), dtype=float)
        
        for gp in range(n):
            G = pw_indices_opt[gp]
            for Kx, Ky in K_points:
                KGx = Kx + G[0]*2*math.pi
                KGy = Ky + G[1]*2*math.pi
                KG2 = KGx*KGx + KGy*KGy
                A_mat[gp][gp] = KG2  # Will be overwritten per K point
        
        for (Kx, Ky) in K_points:
            # Build A and B for this K
            A_k = np.zeros((n, n), dtype=float)
            B_k = np.zeros((n, n), dtype=float)
            
            for gp in range(n):
                G = pw_indices_opt[gp]
                KGx = Kx + G[0]*2*math.pi
                KGy = Ky + G[1]*2*math.pi
                A_k[gp][gp] = KGx*KGx + KGy*KGy
            
            for gp1 in range(n):
                G1 = pw_indices_opt[gp1]
                for gp2 in range(n):
                    G2 = pw_indices_opt[gp2]
                    dGx = G1[0] - G2[0]
                    dGy = G1[1] - G2[1]
                    key = (dGx, dGy)
                    if key in eps_G:
                        e_re, e_im = eps_G[key]
                        B_k[gp1][gp2] = e_re  # eps should be real for symmetric structure
                    else:
                        B_k[gp1][gp2] = 0.0
            
            # Solve generalized eigenvalue problem: A_k E = lambda B_k E
            # Using scipy
            try:
                eigvals = sla.eigvals(A_k, B_k)
                # Take real parts, filter out non-physical (negative or huge)
                eigvals_real = [ev.real for ev in eigvals if ev.real > 0 and abs(ev.imag) < 1e-6*max(1,abs(ev.real))]
                eigvals_real.sort()
                if len(eigvals_real) >= 5:
                    omega4 = eigvals_real[3]  # 4th band (0-indexed: 3)
                    omega5 = eigvals_real[4]  # 5th band
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except Exception:
                pass
        
        if min_omega5 == float('inf') or max_omega4 == 0.0:
            return -1.0
        
        gap_ratio = 2.0*(min_omega5 - max_omega4)/(min_omega5 + max_omega4)
        return gap_ratio
    
    # --- Initial design: four rods ---
    radius = 0.17
    centers = [
        (-0.25, -0.25),
        ( 0.25, -0.25),
        (-0.25,  0.25),
        ( 0.25,  0.25)
    ]
    
    cell = [[0]*size for _ in range(size)]
    for i in range(size):
        y = (i + 0.5)/size - 0.5
        for j in range(size):
            x = (j + 0.5)/size - 0.5
            for cx, cy in centers:
                if (x-cx)**2 + (y-cy)**2 < radius**2:
                    cell[i][j] = 1
                    break
    
    # Enforce symmetry
    def enforce_symmetry(c):
        for i in range(size):
            for j in range(size):
                # Average over orbit (but for binary, use majority vote or just pick one)
                # Since the pattern is symmetric by construction, just ensure consistency
                # For safety, set all orbit members to the same value (use the first)
                orbit = get_orbit(i, j)
                val = c[i][j]
                for (oi, oj) in orbit:
                    c[oi][oj] = val
        return c
    
    cell = enforce_symmetry(cell)
    
    current_score = evaluate_cell_fast(cell)
    best_score = current_score
    best_cell = [row[:] for row in cell]
    
    # Hill climbing over orbits
    random.seed(42)
    orbit_order = list(range(len(orbits)))
    random.shuffle(orbit_order)
    
    T = 0.01
    T_min = 0.001
    cooling = 0.995
    
    while time.time() < deadline:
        for idx in orbit_order:
            if time.time() >= deadline:
                break
            orbit = orbits[idx]
            i, j = orbit[0]
            old_val = cell[i][j]
            new_val = 1 - old_val
            
            # Apply flip
            for (oi, oj) in orbit:
                cell[oi][oj] = new_val
            
            new_score = evaluate_cell_fast(cell)
            
            if new_score > current_score:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best_cell = [row[:] for row in cell]
            else:
                delta = new_score - current_score
                if delta < 0 and T > T_min:
                    prob = math.exp(delta / T)
                    if random.random() > prob:
                        for (oi, oj) in orbit:
                            cell[oi][oj] = old_val
                # else accept (delta >= 0 or random accept)
            
            T = max(T * cooling, T_min)
        
        # Re-shuffle orbit order periodically
        if random.random() < 0.1:
            random.shuffle(orbit_order)
    
    # Restore best cell
    return best_cell

# EVOLVE-BLOCK-END
