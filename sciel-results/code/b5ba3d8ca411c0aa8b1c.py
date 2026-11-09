import numpy as np
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    N, d = X_train.shape
    
    # Standardize X_train for kernel computation
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1.0
    X_train_std = (X_train - X_mean) / X_std
    
    # Standardize y_train for numerical stability
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    y_std[y_std == 0] = 1.0
    y_train_std = (y_train - y_mean) / y_std

    # Hyperparameter grid
    gammas = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    lambdas = [1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]

    # Simple 5-fold CV implementation
    indices = np.arange(N)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = N // 5
    folds = [indices[i:i+fold_size] for i in range(5)]
    # Handle remainder
    if len(indices) > fold_size * 5:
        folds[-1] = indices[fold_size*4:]

    best_score = np.inf
    best_gamma = 1.0
    best_lambda = 1.0

    # Precompute full kernel matrix for each gamma? No, K depends on gamma.
    # But dists is independent of gamma.
    dists = cdist(X_train_std, X_train_std, metric='sqeuclidean')

    for gamma in gammas:
        K = np.exp(-gamma * dists)
        
        for lam in lambdas:
            cv_scores = []
            for i in range(5):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                
                K_train = K[np.ix_(train_idx, train_idx)]
                K_val_train = K[np.ix_(val_idx, train_idx)]
                y_train_fold = y_train_std[train_idx]
                y_val_fold = y_train_std[val_idx]
                
                # Solve (K_train + lam*I) alpha = y_train_fold
                try:
                    # Adding small jitter for stability if needed, but lam should handle it
                    alpha = np.linalg.solve(K_train + lam * np.eye(len(train_idx)), y_train_fold)
                    y_pred_val = K_val_train @ alpha
                    mse = np.mean((y_pred_val - y_val_fold) ** 2)
                    cv_scores.append(mse)
                except np.linalg.LinAlgError:
                    cv_scores.append(np.inf)
            
            mean_cv_score = np.mean(cv_scores)
            if mean_cv_score < best_score:
                best_score = mean_cv_score
                best_gamma = gamma
                best_lambda = lam

    # Final fit with best hyperparameters on all training data
    K = np.exp(-best_gamma * dists)
    A = K + best_lambda * np.eye(N)
    
    try:
        alpha = np.linalg.solve(A, y_train_std)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(A, y_train_std, rcond=None)[0]

    def predict(X):
        X_std = (X - X_mean) / X_std
        dists_test = cdist(X_std, X_train_std, metric='sqeuclidean')
        K_test = np.exp(-best_gamma * dists_test)
        y_pred_std = K_test @ alpha
        y_pred = y_pred_std * y_std + y_mean
        return y_pred

    return predict
