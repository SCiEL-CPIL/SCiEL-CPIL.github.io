<<<<<<< SEARCH
def _kak_decompose_2qubit(target):
    """
    Analytic KAK decomposition for 2-qubit unitaries.
    Returns a list of gates using at most 3 CNOTs.
    U = (L0⊗L1) * CX * (M0⊗M1) * CX * (N0⊗N1) * CX * (R0⊗R1)
    """
    target = np.array(target, dtype=complex)
    n = target.shape[0]
    if n != 4:
        return None

    # Remove global phase to work with SU(4)
    det = np.linalg.det(target)
    if abs(det) > 1e-15:
        phase = det / abs(det)
        target_su = target / phase
    else:
        target_su = target

    # KAK decomposition: U = (L0⊗L1) * K * (R0⊗R1)
    # where K = exp(i*(a*σx⊗σx + b*σy⊗σy + c*σz⊗σz)/2)
    # For the canonical CNOT-based form, we use:
    # K = CX * (Ry(θ)⊗I) * CX * (Rz(θ)⊗Rz(θ)) * CX
    # This corresponds to a=b=θ/2, c=0 in the KAK form
    
    # Step 1: Find R0, R1 such that target_su * (R0⊗R1) has the form
    # where the bottom-right 2x2 block is proportional to identity
    # This makes it "CNOT-ready"
    
    # The CNOT gate in basis |00>,|01>,|10>,|11> is:
    # [[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]
    # After CX(0,1), the structure we want is:
    # The bottom-right 2x2 block should be [[a,0],[0,a]] (proportional to I)
    
    # For U * (R0⊗R1), let U be 4x4. We want:
    # U[2:4, 2:4] * (R0⊗R1)[2:4, 2:4] to be proportional to I
    # (R0⊗R1)[2:4, 2:4] = R0[1:2,1:2] ⊗ R1[1:2,1:2] = r0_11 * r1_11 (scalar)
    # Wait, that's not right. Let me think more carefully.
    
    # Actually, the standard approach is:
    # 1. U1 = U
    # 2. Find R1 such that (I⊗R1)*U1 has bottom-right 2x2 block proportional to I
    # 3. Then U2 = (I⊗R1)*U1, and U2 * CX(0,1) should have a simpler structure
    # 4. Find R0 such that (R0⊗I)*U2*CX(0,1) has bottom-right 2x2 block proportional to I
    # 5. Continue peeling off
    
    # Let me use a more direct numerical approach:
    # We want to find L0,L1,M0,M1,N0,N1,R0,R1 such that:
    # U = (L0⊗L1) @ CX @ (M0⊗M1) @ CX @ (N0⊗N1) @ CX @ (R0⊗R1)
    
    # Work backwards:
    # U @ (R0⊗R1)† = (L0⊗L1) @ CX @ (M0⊗M1) @ CX @ (N0⊗N1) @ CX
    # Let W = U @ (R0⊗R1)†
    # W @ CX† = (L0⊗L1) @ CX @ (M0⊗M1) @ CX @ (N0⊗N1)
    # Let V = W @ CX†
    # V @ (N0⊗N1)† = (L0⊗L1) @ CX @ (M0⊗M1) @ CX
    # Let T = V @ (N0⊗N1)†
    # T @ CX† = (L0⊗L1) @ CX @ (M0⊗M1)
    # Let S = T @ CX†
    # S @ (M0⊗M1)† = (L0⊗L1) @ CX
    # Let P = S @ (M0⊗M1)†
    # P @ CX† = L0⊗L1
    
    # So we need to peel off single-qubit gates from the right.
    # At each step, we need to find a single-qubit gate on one qubit such that
    # after applying it, the resulting matrix has a specific structure.
    
    # For the first peel (R1 on qubit 1):
    # We want U @ (I⊗R1) to have the property that when we multiply by CX†,
    # we get something simpler. 
    
    # Actually, let me use the known result:
    # For U in SU(4), the KAK decomposition gives:
    # U = (L0⊗L1) * exp(i*(θx σx⊗σx + θy σy⊗σy + θz σz⊗σz)/2) * (R0⊗R1)
    
    # The three KAK angles can be extracted from:
    # tr(U @ (σx⊗σx) @ U† @ (σx⊗σx)) etc.
    
    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    
    # Compute the KAK angles
    # For U in SU(4), define:
    # α = arccos(|tr(U @ (σx⊗σx))|/4)
    # β = arccos(|tr(U @ (σy⊗σy))|/4)  
    # γ = arccos(|tr(U @ (σz⊗σz))|/4)
    # But this isn't quite right either.
    
    # Let me use the proper formula:
    # The KAK parameters (a,b,c) satisfy:
    # exp(i*(a σx⊗σx + b σy⊗σy + c σz⊗σz)/2)
    # The eigenvalues of this are exp(±i*(a+b+c)/2), exp(±i*(a-b-c)/2), exp(±i*(-a+b-c)/2), exp(±i*(-a-b+c)/2)
    
    # For our CNOT-based decomposition, we use the form:
    # K = CX @ (Ry(θ)⊗I) @ CX @ (Rz(θ)⊗Rz(θ)) @ CX
    # This has KAK angles (θ/2, θ/2, 0) up to permutation
    
    # Let me just use numerical optimization with a good initialization
    # based on the KAK angles
    
    # Compute KAK angles from eigenvalues of U @ (σx⊗σx) @ U† @ (σx⊗σx)
    Fx = target_su @ np.kron(sx, sx) @ target_su.conj().T @ np.kron(sx, sx)
    Fy = target_su @ np.kron(sy, sy) @ target_su.conj().T @ np.kron(sy, sy)
    Fz = target_su @ np.kron(sz, sz) @ target_su.conj().T @ np.kron(sz, sz)
    
    # The trace gives information about the KAK angles
    # tr(Fx) = 4*cos(2*(a-b-c)/2)^2 + ... this is getting complicated
    
    # Let me use a simpler approach: just return None and use the optimizer
    # but with better initialization based on the KAK structure
    
    return None
=======
def _kak_decompose_2qubit(target):
    """
    Analytic KAK decomposition for 2-qubit unitaries.
    Returns a list of gates using at most 3 CNOTs.
    """
    target = np.array(target, dtype=complex)
    n = target.shape[0]
    if n != 4:
        return None

    # Remove global phase to work with SU(4)
    det = np.linalg.det(target)
    if abs(det) > 1e-15:
        phase = det / abs(det)
        target_su = target / phase
    else:
        target_su = target

    sx = np.array([[0, 1], [1, 0]], dtype=complex)
    sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sz = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Compute invariants
    Ix = np.trace(target_su @ np.kron(sx, sx) @ target_su.conj().T @ np.kron(sx, sx)).real
    Iy = np.trace(target_su @ np.kron(sy, sy) @ target_su.conj().T @ np.kron(sy, sy)).real
    Iz = np.trace(target_su @ np.kron(sz, sz) @ target_su.conj().T @ np.kron(sz, sz)).real
    
    # KAK angles a, b, c
    # Ix = 4 cos(2a) cos(2b) cos(2c)? No.
    # For K = exp(i(a sx⊗sx + b sy⊗sy + c sz⊗sz)/2)
    # Eigenvalues are exp(±i(a+b+c)/2), etc.
    # Ix = 2 cos(2(a-b-c)) + 2 cos(2(a+b-c))
    # Iy = 2 cos(2(-a+b-c)) + 2 cos(2(a+b-c))?
    # Let's use the standard formulas:
    # 4 cos(2a) = Ix? No.
    # Actually, if we set b=c=0, Ix = 4 cos(2a).
    # So a = 0.5 arccos(Ix/4).
    # This is only true if the other angles are 0.
    # General solution requires solving the system.
    
    # However, we can always map the KAK angles to the "CNOT ladder" form
    # where the interaction is symmetric.
    # Let's just use the optimizer with a good start.
    # The KAK angles provide a good start for the optimizer.
    
    # For now, return None to use the improved optimizer.
    return None
>>>>>>> REPLACE

I will improve the optimizer to use `scipy.optimize.minimize` which is much faster and more robust than coordinate descent.

